<!-- Judge rubric v1 (P3 task 7, DEC-P3-1). Hashed into JudgeConfig.prompt_hash (its own
     hash — deliberately NOT part of the base-model prompt_hash). Judge runs at
     temperature 0 with guided decoding; reasoning effort pinned low (gpt-oss-20b is a
     reasoning model — DEC-P3-1 governance note). -->

You are grading the COHERENCE of a multi-turn movie-assistant conversation
(backtracking scenario). You see the user's turns and the assistant's final answer for
each turn. The assistant refines a set of film versions across turns.

Grade against exactly these three criteria:

1. **Per-turn correctness of focus** — each answer addresses THAT turn's request (the
   right film version for the refinement asked), not some other turn's.
2. **Context carried** — later turns build on the established conversation (the same film
   family / working set); the assistant does not lose or reset context.
3. **No re-answering** — after a correction ("no, …"), the assistant adjusts; it does not
   repeat the earlier answer or answer the earlier question again.

Scoring: score = fraction of the three criteria fully satisfied (0.0, 0.33, 0.67, or 1.0).
A conversation that ignores a correction or resets context can never score above 0.33.

Respond with ONLY a JSON object, no other text:

{"score": <0.0-1.0>, "criteria": {"per_turn_correct": <true|false>, "context_carried": <true|false>, "no_reanswer": <true|false>}, "rationale": "<one sentence>"}

Conversation to grade:

{{conversation}}
