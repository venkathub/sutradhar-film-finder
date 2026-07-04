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
    relay.put("BASE_UP")
    log("BASE_UP — waiting for BASE_CAPTURED")
    relay.wait("BASE_CAPTURED", marker_timeout)
    stop(proc)

    # [2] train + merge ----------------------------------------------------------------
    pins = window["training_pips"]
    subprocess.check_call(["pip", "install", "-q", *pins])  # noqa: S603,S607
    config = relay.get("train_config.json")
    train_rows = relay.get("train_rows.jsonl")
    val_rows = relay.get("val_rows.jsonl")
    script = relay.get("train_qlora.py")
    out_dir = Path("/home/ft_out")
    rc = subprocess.call(  # noqa: S603
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--train",
            str(train_rows),
            "--val",
            str(val_rows),
            "--out-dir",
            str(out_dir),
        ]
    )
    metrics = out_dir / "training_metrics.json"
    if metrics.exists():
        relay.put_file(metrics, "out/training_metrics.json")
    if rc != 0:
        raise SystemExit(f"training failed rc={rc}")

    # [3] merged column (byte-identical serving flags) ----------------------------------
    proc = serve_vllm(str(out_dir / "merged"), SERVE_PORT, serve_flags)
    wait_local_health(SERVE_PORT)
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
