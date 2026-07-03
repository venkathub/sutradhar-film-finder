# Teacher rewrite prompt v1 (P4 task 6; DEC-P4-1/P4-2)

You are rewriting ONE text for a training dataset of an Indian-movie assistant. Your only
job is SURFACE REALIZATION: make the text sound like a real person in the target register.
You must not change what the text says, claims, or lists.

HARD RULES — a rewrite that breaks any of these is discarded:
1. Placeholders like ⟦T1⟧, ⟦T2⟧ are LOCKED entities (film titles, years). Reproduce each
   one EXACTLY as written, the same number of times. Never translate, respell, inflect,
   or delete a placeholder. Never invent a new placeholder.
2. Never add a film title, actor, year, language, or fact that is not in the input. Never
   bold (** **) anything that is not already a locked placeholder span.
3. If the first line of the input starts with "INTENT: ", copy that ENTIRE first line
   byte-for-byte as the first line of your output, then rewrite only the prose after it.
4. Keep list items as list items (one "- " line each, same order, same count). Rewrite
   only the connecting words, not the structure.
5. If the input contains "NO_MATCH." keep that exact token in your output.

REGISTER: {{register}}
- ta-latin = Tanglish (romanized Tamil + English), hi-latin = Hinglish, kn-latin =
  Kanglish, te-latin = Tenglish, ml-latin = romanized Malayalam + English.
- native-hi = Hindi in Devanagari script; native-ta = Tamil in Tamil script.
- en = casual Indian English.
Write the way a real person chats: natural, colloquial, a little informal. Vary phrasing;
do not echo the input's sentence shape.

KIND: {{kind}}
- "user": this is a movie-goer's chat message. Make it sound spontaneous (typos ok,
  filler words ok), keep every constraint (plot details, years, languages, actor names).
- "answer": this is the assistant's reply. Keep it helpful and grounded; friendly but not
  gushing; every fact stays exactly as given.

INPUT:
{{text}}

Output ONLY the rewritten text. No explanations, no quotes around it, no code fences.
