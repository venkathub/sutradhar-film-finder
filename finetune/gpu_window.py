"""Self-contained ON-BOX finetune-window driver (P4 task 10; P4_SPEC §2.4; DEC-P2-7).

Runs on the ephemeral GPU box with no repo checkout. Coordinates the strictly-ordered
window with the laptop through HF-relay MARKER files under ``runs/<run_id>/``:

    box:    BASE_UP        (vLLM serving the base model, healthy)
    laptop: BASE_CAPTURED  (base benchmark artifact recorded)
    box:    kill vLLM -> pip-install pinned training stack -> train_qlora.py -> merge
            MERGED_UP      (vLLM serving the MERGED model, byte-identical flags)
    laptop: QLORA_CAPTURED (headline + no-exemplar captures recorded)
    box:    kill vLLM -> serve judge + BGE-M3 embedder
            JUDGE_UP
    laptop: JUDGE_DONE     (both columns re-judged + RAGAS'd)
    box:    upload training_metrics.json + adapter push happened in train step
            EXIT <rc>

Every phase failure still uploads job.log + EXIT so the laptop can diagnose without SSH;
the LAPTOP owns instance teardown (finally-guaranteed) regardless of markers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVE_PORT = 8000
EMBED_PORT = 8001
POLL_S = 15


def log(msg: str) -> None:
    print(f"[gpu-window] {msg}", flush=True)


class Relay:
    def __init__(self, repo: str, run_id: str) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()  # HF_TOKEN from env
        self.repo = repo
        self.run_id = run_id

    def put(self, name: str, payload: bytes = b"1") -> None:
        self.api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=f"runs/{self.run_id}/{name}",
            repo_id=self.repo,
            repo_type="dataset",
        )

    def put_file(self, local: Path, name: str) -> None:
        self.api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f"runs/{self.run_id}/{name}",
            repo_id=self.repo,
            repo_type="dataset",
        )

    def exists(self, name: str) -> bool:
        return bool(
            self.api.file_exists(self.repo, f"runs/{self.run_id}/{name}", repo_type="dataset")
        )

    def exists_abs(self, path_in_repo: str) -> bool:
        return bool(self.api.file_exists(self.repo, path_in_repo, repo_type="dataset"))

    def put_dir(self, local: Path, name: str) -> None:
        self.api.upload_folder(
            folder_path=str(local),
            path_in_repo=f"runs/{self.run_id}/{name}",
            repo_id=self.repo,
            repo_type="dataset",
        )

    def get(self, name: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.repo,
                repo_type="dataset",
                filename=f"runs/{self.run_id}/{name}",
            )
        )

    def wait(self, name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.exists(name):
                return
            time.sleep(POLL_S)
        raise TimeoutError(f"marker {name} not seen within {timeout_s}s")


def wait_local_health(port: int, timeout_s: float = 2700) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001,S110 — poll loop
            pass
        time.sleep(POLL_S)
    raise TimeoutError(f"local vLLM on :{port} not healthy within {timeout_s}s")


PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_title",
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }
]


def tools_selftest(model: str, port: int) -> None:
    """A tools-bearing chat request MUST return 200 before any marker goes up —
    the 2026-07-04 window burn: tools requests 400 without the family tool parser."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "find Pokiri versions"}],
            "tools": PROBE_TOOLS,
            "max_tokens": 32,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise SystemExit(f"tools self-test HTTP {resp.status} — check serve_flags")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode(errors="replace")
        raise SystemExit(
            f"tools self-test FAILED ({exc.code}): {detail} — serve_flags missing the "
            "tool parser? refusing to expose a capture endpoint"
        ) from exc
    log("tools self-test: 200 OK")


def serve_vllm(model: str, port: int, extra: list[str] | None = None) -> subprocess.Popen:
    cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(port), *(extra or [])]
    log(f"serving: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd, stdout=open(f"/root/vllm_{port}.log", "wb"), stderr=subprocess.STDOUT
    )  # noqa: S603,SIM115


def stop(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(10)  # let CUDA memory drain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    relay = Relay(args.repo, args.run_id)

    window = json.loads(relay.get("window_config.json").read_text(encoding="utf-8"))
    base_model = window["base_model"]
    judge_model = window["judge_model"]
    embed_model = window["embed_model"]
    serve_flags: list[str] = window.get("serve_flags", [])
    marker_timeout = float(window.get("marker_timeout_s", 5400))

    os.environ.setdefault("HF_HOME", "/home/hf_cache")
    Path("/home/hf_cache").mkdir(parents=True, exist_ok=True)

    # [1] base column ------------------------------------------------------------------
    subprocess.check_call(["pip", "install", "-q", "-U", "vllm"])  # noqa: S603,S607
    proc = serve_vllm(base_model, SERVE_PORT, serve_flags)
    wait_local_health(SERVE_PORT)
    tools_selftest(base_model, SERVE_PORT)
    relay.put("BASE_UP")
    log("BASE_UP — waiting for BASE_CAPTURED")
    relay.wait("BASE_CAPTURED", marker_timeout)
    stop(proc)

    # [2] train (RESUMABLE: skip if an adapter already exists on the relay) ------------
    # ISOLATED venv: the serving env belongs to vLLM (its own torch build); installing
    # the training pins into it broke transformers' torch-backed lazy imports on the
    # 2026-07-04 attempt. The pins get a clean interpreter; vLLM's env stays untouched
    # for phase [3].
    pins = window["training_pips"]
    venv = Path("/home/trainenv")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])  # noqa: S603
    train_python = str(venv / "bin" / "python")
    subprocess.check_call(  # noqa: S603
        [train_python, "-m", "pip", "install", "-q", "huggingface_hub", *pins]
    )
    out_dir = Path("/home/ft_out")
    adapter_dir = out_dir / "adapter"
    # PHASE CHECKPOINT (2026-07-04 lesson: a merge-serve crash must never cost the
    # training run again). resume_from = a relay prefix holding a finished adapter
    # (e.g. "rescue" or "runs/<old-id>/out"); this run's own out/adapter counts too.
    resume_prefix = window.get("resume_from") or f"runs/{args.run_id}/out"
    if relay.exists_abs(f"{resume_prefix}/adapter/adapter_model.safetensors"):
        log(f"RESUME: adapter found at {resume_prefix}/adapter — skipping training")
        import shutil

        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            allow_patterns=[
                f"{resume_prefix}/adapter/*",
                f"{resume_prefix}/training_metrics.json",
            ],
        )
        shutil.copytree(Path(snap) / resume_prefix / "adapter", adapter_dir, dirs_exist_ok=True)
        tm = Path(snap) / resume_prefix / "training_metrics.json"
        if tm.exists():
            relay.put_file(tm, "out/training_metrics.json")
    else:
        config = relay.get("train_config.json")
        train_rows = relay.get("train_rows.jsonl")
        val_rows = relay.get("val_rows.jsonl")
        script = relay.get("train_qlora.py")
        template = relay.get("gemma4_train_template.jinja")
        rc = subprocess.call(  # noqa: S603
            [
                train_python,
                str(script),
                "--config",
                str(config),
                "--train",
                str(train_rows),
                "--val",
                str(val_rows),
                "--out-dir",
                str(out_dir),
                "--chat-template",
                str(template),
            ]
        )
        metrics = out_dir / "training_metrics.json"
        if metrics.exists():
            relay.put_file(metrics, "out/training_metrics.json")
        if rc != 0:
            raise SystemExit(f"training failed rc={rc}")
        # CHECKPOINT the adapter to the relay IMMEDIATELY — before any serving step can
        # fail — so no later phase can ever cost the training run again.
        relay.put_dir(adapter_dir, "out/adapter")
        log("adapter checkpointed to the relay")

    # [2b] merge: the proven graft recipe (multimodal class + KV-sharing tensors);
    # the processor is saved with the SERVING env python (torchvision/PIL live there).
    merge_script = relay.get("merge_adapter.py")
    merged_dir = out_dir / "merged"
    rc = subprocess.call(  # noqa: S603
        [
            train_python,
            str(merge_script),
            "--base",
            base_model,
            "--adapter",
            str(adapter_dir),
            "--out",
            str(merged_dir),
        ]
    )
    if rc != 0:
        raise SystemExit(f"merge failed rc={rc}")
    rc = subprocess.call(  # noqa: S603
        [
            sys.executable,
            "-c",
            "from transformers import AutoProcessor; "
            f"AutoProcessor.from_pretrained({base_model!r}).save_pretrained({str(merged_dir)!r}); "
            "print('[merge] processor saved')",
        ]
    )
    if rc != 0:
        raise SystemExit("processor save failed")

    # [3] merged column (byte-identical serving flags) ----------------------------------
    proc = serve_vllm(str(merged_dir), SERVE_PORT, serve_flags)
    wait_local_health(SERVE_PORT)
    tools_selftest(str(merged_dir), SERVE_PORT)
    relay.put("MERGED_UP")
    log("MERGED_UP — waiting for QLORA_CAPTURED")
    relay.wait("QLORA_CAPTURED", marker_timeout)
    stop(proc)

    # [5] judge + embedder --------------------------------------------------------------
    embed_proc = subprocess.Popen(  # noqa: S603
        [
            "vllm",
            "serve",
            embed_model,
            "--task",
            "embed",
            "--host",
            "0.0.0.0",
            "--port",
            str(EMBED_PORT),
        ],
        stdout=open("/root/vllm_embed.log", "wb"),  # noqa: SIM115
        stderr=subprocess.STDOUT,
    )
    proc = serve_vllm(judge_model, SERVE_PORT)
    wait_local_health(SERVE_PORT)
    relay.put("JUDGE_UP")
    log("JUDGE_UP — waiting for JUDGE_DONE")
    relay.wait("JUDGE_DONE", marker_timeout)
    stop(proc)
    stop(embed_proc)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        pass  # EXIT marker + job.log upload handled by the startup wrapper
