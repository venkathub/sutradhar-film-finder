// P7 task 4 (DEC-P7-2): live-demo bearer-token adoption + attachment.
// Browser-mode vitest (same runner as App.test.tsx): real sessionStorage/history.

import { beforeEach, describe, expect, it, vi } from "vitest";

import { adoptTokenFromUrl, httpApi } from "./api";

const TOKEN_KEY = "sutradhar_api_token";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  window.sessionStorage.removeItem(TOKEN_KEY);
  window.history.replaceState(null, "", "/");
});

describe("adoptTokenFromUrl", () => {
  it("persists ?token= to sessionStorage and strips it from the URL", () => {
    window.history.replaceState(null, "", "/?token=demo-secret&x=1");
    adoptTokenFromUrl();
    expect(window.sessionStorage.getItem(TOKEN_KEY)).toBe("demo-secret");
    expect(window.location.search).toBe("?x=1"); // token gone, other params kept
  });

  it("is a no-op without a token param", () => {
    adoptTokenFromUrl();
    expect(window.sessionStorage.getItem(TOKEN_KEY)).toBeNull();
  });
});

describe("postChat auth header", () => {
  it("attaches Authorization: Bearer when a token was adopted", async () => {
    window.sessionStorage.setItem(TOKEN_KEY, "demo-secret");
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ status: "off" }));
    await httpApi.postChat({ conversation_id: null, message: "papanasam?" });
    const headers = (fetchSpy.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers.authorization).toBe("Bearer demo-secret");
    fetchSpy.mockRestore();
  });

  it("sends no Authorization header without a token (GPU-off path stays open)", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse({ status: "off" }));
    await httpApi.postChat({ conversation_id: null, message: "papanasam?" });
    const headers = (fetchSpy.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers.authorization).toBeUndefined();
    fetchSpy.mockRestore();
  });
});
