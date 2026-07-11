// Shared test stubs (P6): components take an injected Api, so tests stub the
// four endpoints with plain objects — no fetch mocking, no network.
import type {
  Api,
  ChatUp,
  Replay,
  ReplayList,
  ReplayTurn,
  StatusResponse,
  VersionPayload,
} from "../lib/api";
import type { TurnView } from "../lib/turns";

export function version(overrides: Partial<VersionPayload>): VersionPayload {
  return {
    version_id: "v-0000",
    title: "Drishyam",
    language: "ml",
    year: 2013,
    relationship: "is_original_of",
    is_original: true,
    cast_lead: ["Mohanlal"],
    sources: [{ source: "wikidata", ref: "Q15401703" }],
    confidence: "HIGH",
    ...overrides,
  };
}

/** The GS-01 gate set: the Malayalam original + all four Indian remakes. */
export const DRISHYAM_SET: VersionPayload[] = [
  version({}),
  version({
    version_id: "v-te",
    title: "Drushyam",
    language: "te",
    year: 2014,
    relationship: "is_remake_of",
    is_original: false,
    cast_lead: ["Venkatesh"],
  }),
  version({
    version_id: "v-kn",
    title: "Drishya",
    language: "kn",
    year: 2014,
    relationship: "is_remake_of",
    is_original: false,
    cast_lead: ["Ravichandran"],
  }),
  version({
    version_id: "v-ta",
    title: "Papanasam",
    language: "ta",
    year: 2015,
    relationship: "is_remake_of",
    is_original: false,
    cast_lead: ["Kamal Haasan"],
  }),
  version({
    version_id: "v-hi",
    title: "Drishyam (Hindi)",
    language: "hi",
    year: 2015,
    relationship: "is_remake_of",
    is_original: false,
    cast_lead: ["Ajay Devgn"],
  }),
];

export function turnView(overrides: Partial<TurnView>): TurnView {
  return {
    user: "which movie is papanasam a remake of?",
    answer: 'INTENT: {"intent": "list_versions", "slots": {}}\nFive versions.',
    intent: { intent: "list_versions", slots: {} },
    versions: [],
    citations: [],
    warnings: [],
    trace: [],
    latencyMs: 100,
    replayed: false,
    ...overrides,
  };
}

export const OFF_STATUS: StatusResponse = {
  status: "off",
  detail: "Live demo offline by design — the GPU is on-demand. (endpoint off)",
  evidence: {
    benchmarks: "docs/BENCHMARKS.md",
    replay: "/api/replay/GS-08a",
  },
};

export const UP_STATUS: StatusResponse = {
  status: "up",
  detail: "up",
};

export const REPLAY_LIST: ReplayList = {
  run_id: "20260704T093206Z-e9598564",
  mode: "live",
  model: "google/gemma-4-E4B-it",
  prompt_hash: "78215ccc0000",
  available: ["GS-01", "GS-08a"],
};

export function replayTurn(overrides: Partial<ReplayTurn>): ReplayTurn {
  return {
    message: "the Drishyam with Ajay Devgn",
    answer: 'INTENT: {"intent": "disambiguate", "slots": {}}\nWhich Drishyam?',
    intent: { intent: "disambiguate", slots: {} },
    versions: [],
    citations: [],
    warnings: [],
    latency_ms: 1701.87,
    tool_calls: 1,
    trace: [],
    ...overrides,
  };
}

export const GS08A_REPLAY: Replay = {
  fixture_id: "GS-08a",
  run_id: REPLAY_LIST.run_id,
  mode: "live",
  model: REPLAY_LIST.model,
  prompt_hash: REPLAY_LIST.prompt_hash,
  chat_status: "up",
  turns: [
    replayTurn({}),
    replayTurn({
      message: "no, the original one",
      answer:
        'INTENT: {"intent": "list_versions", "slots": {}}\n' +
        "The original is Drishyam (2013, Malayalam).",
      intent: { intent: "list_versions", slots: {} },
      latency_ms: 3758.77,
    }),
  ],
};

export function chatUp(overrides: Partial<ChatUp>): ChatUp {
  return {
    conversation_id: "conv-1",
    status: "up",
    answer: 'INTENT: {"intent": "find_movie", "slots": {}}\nFound it.',
    intent: { intent: "find_movie", slots: {} },
    versions: [],
    citations: [],
    warnings: [],
    usage: { prompt_tokens: 10, completion_tokens: 5, cost_usd: null },
    latency_ms: 42.0,
    tool_calls: 1,
    trace: [],
    trace_id: null,
    ...overrides,
  };
}

const unexpected = (name: string) => () =>
  Promise.reject(new Error(`unexpected api call: ${name}`));

export function stubApi(overrides: Partial<Api>): Api {
  return {
    getStatus: unexpected("getStatus"),
    postChat: unexpected("postChat"),
    getReplays: unexpected("getReplays"),
    getReplay: unexpected("getReplay"),
    ...overrides,
  };
}
