// Typed client for the P5/P6 API (P6 task 3). Types mirror
// sutradhar.serving.schemas + the degrade payloads — field names unchanged.
// The Api interface is injectable into components (props), so tests stub it
// without any fetch-mocking framework.

export interface SourceRef {
  source: string;
  ref?: string | null;
  [key: string]: unknown;
}

export interface VersionPayload {
  version_id: string;
  title: string;
  language: string | null;
  year: number | null;
  relationship: string | null;
  is_original: boolean;
  cast_lead: string[];
  sources: SourceRef[];
  confidence: string | null;
}

export interface Citation {
  claim_ref: string;
  sources: SourceRef[];
}

export interface TraceStep {
  step: number;
  tool: string;
  arguments: Record<string, unknown> | null;
  valid: boolean;
  validation_error: string | null;
  result_summary: Record<string, unknown>;
  latency_ms: number;
}

export interface IntentPayload {
  intent: string;
  slots: Record<string, unknown>;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number | null;
}

export interface ChatUp {
  conversation_id: string;
  status: "up";
  answer: string;
  intent: IntentPayload | null;
  versions: VersionPayload[];
  citations: Citation[];
  warnings: string[];
  usage: Usage;
  latency_ms: number;
  tool_calls: number;
  trace: TraceStep[];
  trace_id: string | null;
}

export interface OfflineEvidence {
  benchmarks: string;
  replay: string;
  demo_video?: string; // key OMITTED when DEMO_VIDEO_URL is unset — never a dead link
}

export interface ChatOff {
  conversation_id: string | null;
  status: "off";
  detail: string;
  evidence: OfflineEvidence;
  request_live_demo: string;
}

export type ChatResult = ChatUp | ChatOff;

export interface ChatError {
  error: string;
  detail: string;
}

export interface StatusResponse {
  status: "up" | "off" | "error";
  detail: string;
  evidence?: OfflineEvidence;
}

export interface ReplayTurn {
  message: string;
  answer: string;
  intent: IntentPayload | null;
  versions: VersionPayload[];
  citations: Citation[];
  warnings: string[];
  latency_ms: number;
  tool_calls: number;
  trace: TraceStep[];
}

export interface ReplayList {
  run_id: string;
  mode: string;
  model: string;
  prompt_hash: string;
  available: string[];
}

export interface Replay {
  fixture_id: string;
  run_id: string;
  mode: string;
  model: string;
  prompt_hash: string;
  chat_status: string;
  turns: ReplayTurn[];
}

export interface Api {
  getStatus(): Promise<StatusResponse>;
  postChat(body: {
    conversation_id: string | null;
    message: string;
  }): Promise<ChatResult | ChatError>;
  getReplays(): Promise<ReplayList>;
  getReplay(fixtureId: string): Promise<Replay>;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} -> HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

// Same-origin paths only: the built UI is served by the API itself (task 2),
// and the dev server proxies /api (vite.config.ts). No base URL, no CORS.
export const httpApi: Api = {
  getStatus: () => getJson<StatusResponse>("/api/status"),
  postChat: async (body) => {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    // 200 = up or structured off; 4xx = structured {error, detail} — both render.
    return (await response.json()) as ChatResult | ChatError;
  },
  getReplays: () => getJson<ReplayList>("/api/replays"),
  getReplay: (fixtureId) =>
    getJson<Replay>(`/api/replay/${encodeURIComponent(fixtureId)}`),
};
