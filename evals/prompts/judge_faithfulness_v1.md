<!-- Judge rubric v1 (P3 task 7, DEC-P3-1). Hashed into JudgeConfig.prompt_hash. Used for
     the human-agreement validation sample (judge-vs-human κ); the GATING faithfulness
     signal stays the deterministic no-hallucinated-movie detector (DEC-P3-3). -->

You are grading the FAITHFULNESS of a movie-assistant answer against the tool results the
assistant received. The tool results are the ONLY permitted source of film facts.

Grade against exactly these two criteria:

1. **No invented films** — every film the answer asserts (title/year/language) appears in
   the tool-result film list below. Minor spelling/transliteration variants of a listed
   film are NOT inventions.
2. **No contradicted facts** — years, languages, relationships (original / remake / dub)
   stated in the answer match the tool results; an abstaining answer ("not in the
   catalog") is faithful when the tool results are empty.

Scoring: score = 1.0 if both criteria hold, 0.5 if films are real but a fact is
contradicted, 0.0 if any film is invented.

Respond with ONLY a JSON object, no other text:

{"score": <0.0|0.5|1.0>, "criteria": {"no_invented_films": <true|false>, "no_contradicted_facts": <true|false>}, "rationale": "<one sentence>"}

Tool-result film list:

{{allowed_titles}}

Answer to grade:

{{answer}}
