"""Deterministic scaffold generator (P4 task 4, P4_SPEC §2.2/§2.3; DEC-P4-2).

A PURE function of ``(ScaffoldSnapshot, ScaffoldConfig)``: seeded sampling of gate-view
recordings → behaviour-classed conversation skeletons with v0 tool-call sequences,
ground-truth tool results, and labelled answers. Properties held by construction (and
re-asserted by Tier-1 tests):

- **Grounding:** every tool result is a recorded repository ``model_dump`` (or, for
  ``search_by_plot``/``refine_filter``, deterministically constructed from those recorded
  rows) — every asserted title in every answer resolves to that conversation's own tool
  results, so *no invented films are trainable*.
- **Schema:** tool names/arguments are emitted against ``tool_schema.v0.json`` — the
  generator has no other vocabulary.
- **Contracts:** the final prose answer of each user turn starts with exactly one INTENT
  preamble line; asserted titles are ``**bold**`` and nothing else is bold; abstentions
  end with ``NO_MATCH.`` and assert zero titles (mirrors the frozen exemplar format).
- **No system turns:** the frozen prompt bundle is attached at render time (task 8), so
  the same conversations serve both D6 prompt variants.

Scaffold *surfaces* (user utterances, answer prose) are deliberately plain templates —
register-rich code-mix is the teacher pass's job (D2); entities stay literal here and get
sentinel-locked for the teacher (task 6).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sutradhar.finetune.dataset import ToolCallRecord, TrainingConversation, TrainingMessage
from sutradhar.finetune.snapshot import ScaffoldSnapshot, WorkSnapshot

# --- Configuration (spec §2.2 mix table) ---

BEHAVIOUR_SHARES: dict[str, float] = {
    "find_by_plot": 0.30,
    "find_by_title": 0.15,
    "list_versions": 0.10,
    "refine": 0.25,
    "disambiguate": 0.05,
    "out_of_catalog": 0.15,
}

CODE_MIXED_LANGS = ("ta-latin", "hi-latin", "kn-latin", "te-latin", "ml-latin")


class ScaffoldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    size: int = 2000
    behaviour_shares: dict[str, float] = Field(default_factory=lambda: dict(BEHAVIOUR_SHARES))
    code_mixed_share: float = 0.45  # spec: >= 0.40
    native_share: float = 0.15  # spec: >= 0.10 (remainder = en)


# --- Deterministic RNG (stdlib-independent so hash stability survives Python upgrades) ---


class _Rng:
    """Tiny SplitMix64 — deterministic across platforms/Python versions by construction."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def random(self) -> float:
        return self._next() / 2**64

    def randint(self, low: int, high: int) -> int:  # inclusive
        return low + self._next() % (high - low + 1)

    def choice(self, seq: list[Any]) -> Any:
        return seq[self._next() % len(seq)]

    def shuffle(self, seq: list[Any]) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self._next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.random()


# --- Shared prose helpers ---

LANGUAGE_NAMES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "bn": "Bengali",
    "si": "Sinhala",
    "zh": "Chinese",
    "en": "English",
}

REL_PHRASES = {
    "is_original_of": "original",
    "is_remake_of": "remake of the original",
    "is_official_dub_of": "official dub",
    "is_unofficial_remake_of": "unofficial remake",
    "is_sequel_of": "sequel",
    None: "relationship unverified",
}

_SOURCE_LABELS = {"wikidata": "Wikidata", "tmdb": "TMDB", "imdb": "IMDb", "human": "curated"}


def _cite(sources: list[dict[str, Any]]) -> str | None:
    if not sources:
        return None
    # Prefer structured sources over the curated-seed provenance row for the visible cite.
    ranked = sorted(
        sources, key=lambda s: {"wikidata": 0, "tmdb": 1, "imdb": 2}.get(str(s.get("source")), 9)
    )
    src = ranked[0]
    label = _SOURCE_LABELS.get(str(src.get("source")), str(src.get("source")))
    return f"{label} {src.get('ref')}"


def _preamble(intent: str, slots: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"intent": intent}
    if slots:
        payload["slots"] = slots
    return "INTENT: " + json.dumps(payload, ensure_ascii=False)


def _version_line(entry: dict[str, Any]) -> str:
    lang = LANGUAGE_NAMES.get(entry.get("language") or "", entry.get("language") or "?")
    lead = entry.get("cast_lead") or []
    rel = REL_PHRASES.get(entry.get("relationship"), str(entry.get("relationship")))
    parts = [f"- **{entry['title']}** ({entry.get('year')}, {lang})"]
    if lead:
        parts.append(lead[0])
    parts.append(rel)
    cite = _cite(entry.get("sources") or [])
    if cite:
        parts.append(f"({cite})")
    return " — ".join(parts)


def _answer_block(preamble: str, headline: str, lines: list[str], footer: str | None = None) -> str:
    body = "\n".join([headline, *lines])
    if footer:
        body = f"{body}\n{footer}"
    return f"{preamble}\n\n{body}"


def _mask_titles(text: str, titles: list[str]) -> str:
    masked = text
    for title in sorted(titles, key=len, reverse=True):
        masked = re.sub(re.escape(title), "the film", masked, flags=re.IGNORECASE)
    return masked


# --- Template banks (scaffold register — the teacher enriches these, D2) ---
# Keyed by language family; "native-hi"/"native-ta" are the native-script banks.

USER_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "find_by_plot": {
        "en": [
            "looking for a movie where {desc}",
            "what's that film where {desc}?",
            "trying to remember a movie — {desc}",
        ],
        "hi-latin": [
            "yaar wo movie kaunsi hai jisme {desc}",
            "ek film thi jisme {desc} — naam yaad nahi aa raha",
        ],
        "ta-latin": [
            "oru padam la {desc} — enna movie adhu?",
            "andha padam peru theriyala, adhula {desc}",
        ],
        "kn-latin": [
            "ondu film alli {desc} — yavudu adu?",
            "aa cinema nenapagtilla, adralli {desc}",
        ],
        "te-latin": [
            "oka cinema lo {desc} — ye movie adi?",
            "aa movie peru gurthu ledu, andulo {desc}",
        ],
        "ml-latin": [
            "oru padathil {desc} — ethu padam aanu?",
            "aa cinema perinu orma illa, athil {desc}",
        ],
        "native-hi": ["वो फिल्म कौनसी है जिसमें {desc}?", "एक फिल्म थी जिसमें {desc} — नाम बताओ"],
        "native-ta": ["அந்த படத்தில் {desc} — எந்த படம்?", "ஒரு படம் — {desc}. பெயர் சொல்லுங்க"],
    },
    "find_by_title": {
        "en": [
            "is {title} a remake? which one is the original?",
            "tell me about {title} — original or remake?",
        ],
        "hi-latin": [
            "{title} remake hai kya? original kaunsa hai",
            "yaar {title} ka original version kaunsa hai?",
        ],
        "ta-latin": [
            "{title} remake ah? original edhu?",
            "{title} padam original ah illa remake ah?",
        ],
        "kn-latin": ["{title} remake na? original yavudu?"],
        "te-latin": ["{title} remake aa? original edi?"],
        "ml-latin": ["{title} remake aano? original ethu?"],
        "native-hi": ["क्या {title} रीमेक है? ओरिजिनल कौनसी है?"],
        "native-ta": ["{title} ரீமேக்கா? ஒரிஜினல் எது?"],
    },
    "list_versions": {
        "en": [
            "which languages was {title} made in? give me the full list",
            "list every version of {title} in the catalog",
        ],
        "hi-latin": [
            "{title} ka har language version batao",
            "{title} kitni languages mein bana hai?",
        ],
        "ta-latin": ["{title} oda ella language versions um sollu"],
        "kn-latin": ["{title} yav yav bhashegalalli ide? full list kodi"],
        "te-latin": ["{title} anni bhashallo unda? full list ivvu"],
        "ml-latin": ["{title} ethra bhashakalil undu? full list tharu"],
        "native-hi": ["{title} के सारे संस्करण दिखाओ"],
        "native-ta": ["{title} எல்லா மொழி பதிப்புகளையும் காட்டு"],
    },
    "list_versions_sequels": {
        "en": ["walk me through the whole {title} franchise, sequels included"],
        "hi-latin": ["{title} ka poora franchise batao, sequels bhi"],
        "ta-latin": ["{title} full franchise sollu, sequels um serthu"],
        "kn-latin": ["{title} full franchise heli, sequels jothege"],
        "te-latin": ["{title} full franchise cheppu, sequels tho kalipi"],
        "ml-latin": ["{title} full franchise parayu, sequels adakkam"],
        "native-hi": ["{title} की पूरी फ्रैंचाइज़ बताओ, सीक्वल भी"],
        "native-ta": ["{title} முழு தொடரையும் சொல்லு, தொடர்ச்சிகளும் சேர்த்து"],
    },
    "out_of_catalog": {
        "en": ["any movie like this: {desc}?", "{desc} — does this film exist?"],
        "hi-latin": ["{desc} — aisi koi movie hai kya?"],
        "ta-latin": ["{desc} — ipdi oru padam irukka?"],
        "kn-latin": ["{desc} — ee tara film ideya?"],
        "te-latin": ["{desc} — ilanti cinema unda?"],
        "ml-latin": ["{desc} — ingane oru padam undo?"],
        "native-hi": ["{desc} — ऐसी कोई फिल्म है क्या?"],
        "native-ta": ["{desc} — இப்படி ஒரு படம் இருக்கா?"],
    },
}

REFINE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "language": {
        "en": ["only the {value} one please", "just show the {value} version"],
        "hi-latin": ["mujhe bas {value} version do", "bas {value} wala dikhao"],
        "ta-latin": ["{value} version mattum kaamí", "{value} la irukkuradhu mattum"],
        "kn-latin": ["{value} version matra thorisi"],
        "te-latin": ["{value} version matrame chupinchu"],
        "ml-latin": ["{value} version mathram kaanikku"],
        "native-hi": ["सिर्फ {value} वाला दिखाओ"],
        "native-ta": ["{value} பதிப்பு மட்டும் காட்டு"],
    },
    "year": {
        "en": ["the {value} one", "which one came out in {value}?"],
        "hi-latin": ["{value} wali kaunsi thi?"],
        "ta-latin": ["{value} la vandhadhu edhu?"],
        "kn-latin": ["{value} nalli bandiddu yavudu?"],
        "te-latin": ["{value} lo vachindi edi?"],
        "ml-latin": ["{value} il vannathu ethu?"],
        "native-hi": ["{value} वाली कौनसी थी?"],
        "native-ta": ["{value} இல் வந்தது எது?"],
    },
    "actor": {
        "en": ["the one with {value}", "I mean the {value} film"],
        "hi-latin": ["{value} wali movie yaar"],
        "ta-latin": ["{value} nadicha padam"],
        "kn-latin": ["{value} madida film"],
        "te-latin": ["{value} chesina cinema"],
        "ml-latin": ["{value} cheytha padam"],
        "native-hi": ["{value} वाली फिल्म"],
        "native-ta": ["{value} நடித்த படம்"],
    },
    "era": {
        "en": ["something newer than the original", "an older one actually"],
        "hi-latin": ["original se naya wala chahiye", "nahi, purana wala"],
        "ta-latin": ["original ku apparam vandhadhu", "illa, pazhaya version"],
        "kn-latin": ["original nantara bandiddu", "illa, haleyadu"],
        "te-latin": ["original tarvata vachindi", "kadu, pathadi"],
        "ml-latin": ["original nu shesham vannathu", "alla, pazhayathu"],
        "native-hi": ["ओरिजिनल से नया वाला", "नहीं, पुराना वाला"],
        "native-ta": ["ஒரிஜினலுக்கு பிறகு வந்தது", "இல்ல, பழைய பதிப்பு"],
    },
}

HEADLINES: dict[str, str] = {
    "en": "Found it — that's **{title}**. {n} version(s) in the catalog:",
    "hi-latin": "Ye **{title}** hai! Catalog mein {n} version(s) hain:",
    "ta-latin": "Idhu **{title}** padam! Catalog la {n} version(s) irukku:",
    "kn-latin": "Idu **{title}** film! Catalog alli {n} version(s) ive:",
    "te-latin": "Idi **{title}** cinema! Catalog lo {n} version(s) unnayi:",
    "ml-latin": "Ithu **{title}** aanu! Catalog il {n} version(s) undu:",
    "native-hi": "यह **{title}** है! कैटलॉग में {n} संस्करण हैं:",
    "native-ta": "இது **{title}**! கேட்டலாகில் {n} பதிப்புகள் உள்ளன:",
}

REFINE_HEADLINES: dict[str, str] = {
    "en": "Narrowed down — {n} match(es):",
    "hi-latin": "Filter kar diya — {n} match(es):",
    "ta-latin": "Filter pannitten — {n} match(es):",
    "kn-latin": "Filter maadide — {n} match(es):",
    "te-latin": "Filter chesanu — {n} match(es):",
    "ml-latin": "Filter cheythu — {n} match(es):",
    "native-hi": "फ़िल्टर कर दिया — {n} मिले:",
    "native-ta": "வடிகட்டிவிட்டேன் — {n} பொருத்தம்:",
}

REFINE_EMPTY: dict[str, str] = {
    "en": "None of the versions in this set match that filter. Want me to relax it?",
    "hi-latin": "Is set mein aisa koi version nahi mila. Filter thoda relax karun?",
    "ta-latin": "Indha set la appadi version illa. Filter konjam relax pannattuma?",
    "kn-latin": "Ee set alli antha version illa. Filter relax maadala?",
    "te-latin": "Ee set lo alanti version ledu. Filter relax cheyyala?",
    "ml-latin": "Ee set il angane oru version illa. Filter relax cheyyatte?",
    "native-hi": "इस सेट में ऐसा कोई संस्करण नहीं मिला। फ़िल्टर थोड़ा ढीला करूँ?",
    "native-ta": "இந்த செட்டில் அப்படி பதிப்பு இல்லை. வடிகட்டியை தளர்த்தட்டுமா?",
}

ABSTAIN: dict[str, str] = {
    "en": (
        "I checked the catalog by story and found nothing that matches — I won't guess a "
        "film that isn't there. NO_MATCH. Want to refine the description?"
    ),
    "hi-latin": (
        "Catalog mein story se check kiya, aisi koi film nahi mili — main andaza nahi "
        "lagaunga. NO_MATCH. Description thoda badal ke try karein?"
    ),
    "ta-latin": (
        "Catalog la story vachu paathen, appadi oru padam illa — naan guess panna "
        "maatten. NO_MATCH. Description konjam maathi try pannalaama?"
    ),
    "kn-latin": (
        "Catalog alli story inda nodide, antha film illa — naanu guess maadalla. "
        "NO_MATCH. Description swalpa badalisi try maadona?"
    ),
    "te-latin": (
        "Catalog lo story tho chusanu, alanti cinema ledu — nenu guess cheyyanu. "
        "NO_MATCH. Description konchem marchi try cheddama?"
    ),
    "ml-latin": (
        "Catalog il story vechu nokki, angane oru padam illa — njan guess cheyyilla. "
        "NO_MATCH. Description onnu maatti nokkatte?"
    ),
    "native-hi": (
        "कैटलॉग में कहानी से देखा, ऐसी कोई फिल्म नहीं मिली — मैं अनुमान नहीं लगाऊँगा। "
        "NO_MATCH. विवरण बदलकर कोशिश करें?"
    ),
    "native-ta": (
        "கேட்டலாகில் கதை வைத்து பார்த்தேன், அப்படி படம் இல்லை — நான் யூகிக்க மாட்டேன். "
        "NO_MATCH. விளக்கத்தை மாற்றி முயற்சிக்கலாமா?"
    ),
}

ASK_BACK: dict[str, str] = {
    "en": "The title {title} spans more than one film — which one do you mean: {options}?",
    "hi-latin": "{title} naam ki ek se zyada filmein hain — kaunsi wali: {options}?",
    "ta-latin": "{title} nu onnukku mela padam irukku — edhu venum: {options}?",
    "kn-latin": "{title} hesarina ondakkinta hechu film ive — yavudu beku: {options}?",
    "te-latin": "{title} pero okati kante ekkuva cinemalu unnayi — edi kavali: {options}?",
    "ml-latin": "{title} ennu perulla onnil kooduthal padangal undu — ethu venam: {options}?",
    "native-hi": "{title} नाम की एक से ज़्यादा फिल्में हैं — कौनसी वाली: {options}?",
    "native-ta": "{title} என்ற பெயரில் ஒன்றுக்கு மேல் படங்கள் உள்ளன — எது வேண்டும்: {options}?",
}

PICK_TEMPLATES: dict[str, str] = {
    "en": "the {year} {language} one",
    "hi-latin": "{year} wali {language} movie",
    "ta-latin": "{year} {language} padam",
    "kn-latin": "{year} {language} film",
    "te-latin": "{year} {language} cinema",
    "ml-latin": "{year} {language} padam",
    "native-hi": "{year} वाली {language} फिल्म",
    "native-ta": "{year} {language} படம்",
}


# --- Local refine (mirrors repository.refine_filter over recorded VersionEntry rows) ---


def refine_local(entries: list[dict[str, Any]], by: dict[str, Any]) -> dict[str, Any]:
    original_years = [e["year"] for e in entries if e.get("is_original") and e.get("year")]
    all_years = [e["year"] for e in entries if e.get("year")]
    pivot = min(original_years) if original_years else (min(all_years) if all_years else None)
    kept = []
    for e in entries:
        if by.get("language") is not None and e.get("language") != by["language"]:
            continue
        if by.get("year") is not None and e.get("year") != by["year"]:
            continue
        if by.get("actor") is not None:
            names = " ".join(e.get("cast_lead") or []).casefold()
            if by["actor"].casefold() not in names:
                continue
        if by.get("relationship") is not None and e.get("relationship") != by["relationship"]:
            continue
        era = by.get("era")
        if era is not None:
            year = e.get("year")
            if era == "original" and not e.get("is_original"):
                continue
            if era == "newer" and (pivot is None or year is None or year <= pivot):
                continue
            if era == "older" and (pivot is None or year is None or year >= pivot):
                continue
        kept.append(e)
    return {
        "versions": [
            {
                "version_id": e["version_id"],
                "title": e["title"],
                "language": e["language"],
                "year": e.get("year"),
                "relationship": e.get("relationship"),
                "is_original": bool(e.get("is_original")),
            }
            for e in kept
        ]
    }


# --- The generator ---


def _allocate(shares: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation — exact, deterministic, sums to total."""
    raw = {k: shares[k] * total for k in sorted(shares)}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    by_frac = sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k))
    for k in by_frac[:remainder]:
        counts[k] += 1
    return counts


def _language_plan(config: ScaffoldConfig) -> dict[str, int]:
    shares: dict[str, float] = {
        lang: config.code_mixed_share / len(CODE_MIXED_LANGS) for lang in CODE_MIXED_LANGS
    }
    shares["native"] = config.native_share
    shares["en"] = 1.0 - config.code_mixed_share - config.native_share
    return _allocate(shares, config.size)


class _ConvDraft(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turns: list[TrainingMessage]
    intent_labels: list[str]
    slot_labels: list[dict[str, Any]]
    entity_ids: list[str]
    plan: dict[str, Any]


def _tool_turns(
    tool: str, arguments: dict[str, Any], result: dict[str, Any]
) -> list[TrainingMessage]:
    return [
        TrainingMessage(
            role="assistant", tool_calls=[ToolCallRecord(tool=tool, arguments=arguments)]
        ),
        TrainingMessage(role="tool", tool_result=result),
    ]


def _entity_ids(gv: dict[str, Any], work_id: str) -> list[str]:
    ids = {work_id}
    for e in gv.get("versions", []):
        ids.add(str(e["version_id"]))
    return sorted(ids)


def _plot_search_result(
    rng: _Rng, work: WorkSnapshot, others: list[WorkSnapshot]
) -> dict[str, Any]:
    results = [
        {
            "work_id": work.work_id,
            "canonical_title": work.canonical_title,
            "language": work.original_language,
            "year": work.get_work.get("first_release_year"),
            "score": round(rng.uniform(0.78, 0.93), 4),
        }
    ]
    if others and rng.random() < 0.5:
        distractor = rng.choice(others)
        results.append(
            {
                "work_id": distractor.work_id,
                "canonical_title": distractor.canonical_title,
                "language": distractor.original_language,
                "year": distractor.get_work.get("first_release_year"),
                "score": round(rng.uniform(0.40, 0.55), 4),
            }
        )
    return {"results": results, "abstain": False}


def _plot_description(rng: _Rng, work: WorkSnapshot) -> str:
    excerpt = rng.choice(work.plot_excerpts).excerpt
    titles = [work.canonical_title] + [
        e["title"] for e in work.get_versions["indian"].get("versions", [])
    ]
    desc = _mask_titles(excerpt, titles).strip().rstrip(".")
    return desc[:1].lower() + desc[1:]


def _versions_answer(
    family: str, headline_title: str, gv: dict[str, Any], intent: str, slots: dict[str, Any]
) -> str:
    entries = gv.get("versions", [])
    headline = HEADLINES[family].format(title=headline_title, n=len(entries))
    return _answer_block(_preamble(intent, slots), headline, [_version_line(e) for e in entries])


def _gen_find_by_plot(
    rng: _Rng, works: list[WorkSnapshot], all_works: list[WorkSnapshot], family: str
) -> _ConvDraft:
    work = rng.choice(works)
    desc = _plot_description(rng, work)
    user_text = rng.choice(USER_TEMPLATES["find_by_plot"][family]).format(desc=desc)
    sp_result = _plot_search_result(rng, work, [w for w in all_works if w is not work])
    gv = work.get_versions["indian"]
    slots = {"plot_description": desc}
    turns = [
        TrainingMessage(role="user", content=user_text),
        *_tool_turns("search_by_plot", {"description": desc, "top_k": 10}, sp_result),
        *_tool_turns("get_versions", {"work_id": work.work_id, "scope": "indian"}, gv),
        TrainingMessage(
            role="assistant",
            content=_versions_answer(family, work.canonical_title, gv, "find_by_plot", slots),
        ),
    ]
    return _ConvDraft(
        turns=turns,
        intent_labels=["find_by_plot"],
        slot_labels=[slots],
        entity_ids=_entity_ids(gv, work.work_id),
        plan={"behaviour": "find_by_plot", "work": work.work_key, "desc": desc},
    )


def _unambiguous_queries(work: WorkSnapshot) -> list[str]:
    return sorted(
        q
        for q, res in work.resolve_title.items()
        if res.get("candidates") and not res.get("ambiguous")
    )


def _gen_find_by_title(
    rng: _Rng, works: list[WorkSnapshot], family: str, behaviour: str
) -> _ConvDraft:
    candidates = [w for w in works if _unambiguous_queries(w)]
    work = rng.choice(candidates)
    query = rng.choice(_unambiguous_queries(work))
    rt = work.resolve_title[query]
    top = rt["candidates"][0]
    sequels = behaviour == "list_versions" and "indian_sequels" in work.get_versions
    variant = "indian_sequels" if sequels and rng.random() < 0.6 else "indian"
    gv = work.get_versions[variant]
    gv_args: dict[str, Any] = {"work_id": work.work_id, "scope": "indian"}
    template_key = behaviour
    if variant == "indian_sequels":
        gv_args["include_sequels"] = True
        template_key = "list_versions_sequels"
    user_text = rng.choice(USER_TEMPLATES[template_key][family]).format(title=query)
    slots = {"title": query}
    turns = [
        TrainingMessage(role="user", content=user_text),
        *_tool_turns("resolve_title", {"title": query}, rt),
        *_tool_turns("get_versions", gv_args, gv),
        TrainingMessage(
            role="assistant",
            content=_versions_answer(family, top["matched_title"], gv, behaviour, slots),
        ),
    ]
    return _ConvDraft(
        turns=turns,
        intent_labels=[behaviour],
        slot_labels=[slots],
        entity_ids=_entity_ids(gv, work.work_id),
        plan={"behaviour": behaviour, "work": work.work_key, "query": query, "variant": variant},
    )


def _refine_dimensions(entries: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    dims: list[tuple[str, Any]] = []
    langs = sorted({e["language"] for e in entries})
    for lang in langs:
        if 0 < sum(1 for e in entries if e["language"] == lang) < len(entries):
            dims.append(("language", lang))
    years = sorted({e["year"] for e in entries if e.get("year")})
    for year in years:
        if 0 < sum(1 for e in entries if e.get("year") == year) < len(entries):
            dims.append(("year", year))
    actors = sorted({(e.get("cast_lead") or ["?"])[0] for e in entries if e.get("cast_lead")})
    for actor in actors[:3]:
        dims.append(("actor", actor))
    if any(e.get("is_original") for e in entries) and len(entries) > 1:
        dims.append(("era", "newer"))
    return dims


_ABSENT_LANGS = ("bn", "ml", "kn", "te")


def _gen_refine(rng: _Rng, works: list[WorkSnapshot], family: str) -> _ConvDraft:
    candidates = [
        w
        for w in works
        if len(w.get_versions["indian"].get("versions", [])) >= 3 and _unambiguous_queries(w)
    ]
    work = rng.choice(candidates)
    query = rng.choice(_unambiguous_queries(work))
    rt = work.resolve_title[query]
    gv = work.get_versions["indian"]
    entries = gv.get("versions", [])
    version_set = [str(e["version_id"]) for e in entries]
    base_slots = {"title": query}
    turns = [
        TrainingMessage(
            role="user",
            content=rng.choice(USER_TEMPLATES["list_versions"][family]).format(title=query),
        ),
        *_tool_turns("resolve_title", {"title": query}, rt),
        *_tool_turns("get_versions", {"work_id": work.work_id, "scope": "indian"}, gv),
        TrainingMessage(
            role="assistant",
            content=_versions_answer(
                family, rt["candidates"][0]["matched_title"], gv, "list_versions", base_slots
            ),
        ),
    ]
    intent_labels = ["list_versions"]
    slot_labels: list[dict[str, Any]] = [base_slots]

    dims = _refine_dimensions(entries)
    if not dims:  # defensive: every >=3-version work has a partitioning language dim
        dims = [("language", entries[0]["language"])]
    n_refines = rng.randint(1, min(3, len(dims)))
    empty_turn = rng.random() < 0.25
    plan_dims: list[Any] = []
    for i in range(n_refines):
        is_last = i == n_refines - 1
        if empty_turn and is_last:
            present = {e["language"] for e in entries}
            absent = [lang for lang in _ABSENT_LANGS if lang not in present]
            if absent:
                dim, value = "language", absent[0]
            else:
                empty_turn = False
        if not (empty_turn and is_last):
            dim, value = rng.choice(dims)
        by = {dim: value}
        result = refine_local(entries, by)
        user_value = LANGUAGE_NAMES.get(str(value), str(value)) if dim == "language" else value
        template = REFINE_TEMPLATES[dim][family][0]
        user_text = template.format(value=user_value)
        kept = result["versions"]
        if kept:
            headline = REFINE_HEADLINES[family].format(n=len(kept))
            answer = _answer_block(
                _preamble("refine", by), headline, [_version_line(e) for e in kept]
            )
        else:
            answer = f"{_preamble('refine', by)}\n\n{REFINE_EMPTY[family]}"
        turns.extend(
            [
                TrainingMessage(role="user", content=user_text),
                *_tool_turns("refine_filter", {"version_set": version_set, "by": by}, result),
                TrainingMessage(role="assistant", content=answer),
            ]
        )
        intent_labels.append("refine")
        slot_labels.append(dict(by))
        plan_dims.append([dim, value])
    return _ConvDraft(
        turns=turns,
        intent_labels=intent_labels,
        slot_labels=slot_labels,
        entity_ids=_entity_ids(gv, work.work_id),
        plan={"behaviour": "refine", "work": work.work_key, "query": query, "dims": plan_dims},
    )


def _gen_disambiguate(rng: _Rng, snapshot: ScaffoldSnapshot, family: str) -> _ConvDraft:
    ambiguous: list[tuple[WorkSnapshot, str]] = []
    for work in snapshot.works:
        for q, res in sorted(work.resolve_title.items()):
            if res.get("ambiguous"):
                ambiguous.append((work, q))
    if not ambiguous:
        raise ValueError("no ambiguous resolve_title recording in snapshot (need a collision pair)")
    work, query = rng.choice(ambiguous)
    rt = work.resolve_title[query]
    by_work: dict[str, dict[str, Any]] = {}
    for c in rt["candidates"]:
        by_work.setdefault(str(c["work_id"]), c)
    options = " or ".join(
        "**{t}** ({y}, {lang})".format(
            t=c["matched_title"],
            y=c.get("year"),
            lang=LANGUAGE_NAMES.get(c.get("language") or "", c.get("language")),
        )
        for c in by_work.values()
    )
    ask = ASK_BACK[family].format(title=query, options=options)
    slots1 = {"title": query}
    works_by_id = {w.work_id: w for w in snapshot.works}
    chosen_id = rng.choice(sorted(k for k in by_work if k in works_by_id))
    chosen_work = works_by_id[chosen_id]
    chosen_candidate = by_work[chosen_id]
    gv = chosen_work.get_versions["indian"]
    pick_lang = LANGUAGE_NAMES.get(chosen_candidate.get("language") or "", "?")
    pick = PICK_TEMPLATES[family].format(year=chosen_candidate.get("year"), language=pick_lang)
    # The pick distinguishes by year (always distinguishing across candidate works here);
    # language is included only when it actually separates the options.
    slots2: dict[str, Any] = {"title": query, "year": chosen_candidate.get("year")}
    candidate_langs = {c.get("language") for c in by_work.values()}
    if len(candidate_langs) > 1:
        slots2["language"] = chosen_candidate.get("language")
    turns = [
        TrainingMessage(
            role="user",
            content=rng.choice(USER_TEMPLATES["list_versions"][family]).format(title=query),
        ),
        *_tool_turns("resolve_title", {"title": query}, rt),
        TrainingMessage(role="assistant", content=f"{_preamble('disambiguate', slots1)}\n\n{ask}"),
        TrainingMessage(role="user", content=pick),
        *_tool_turns("get_versions", {"work_id": chosen_id, "scope": "indian"}, gv),
        TrainingMessage(
            role="assistant",
            content=_versions_answer(
                family, chosen_candidate["matched_title"], gv, "list_versions", slots2
            ),
        ),
    ]
    return _ConvDraft(
        turns=turns,
        intent_labels=["disambiguate", "list_versions"],
        slot_labels=[slots1, slots2],
        entity_ids=_entity_ids(gv, chosen_id),
        plan={"behaviour": "disambiguate", "query": query, "chosen": chosen_work.work_key},
    )


def _gen_out_of_catalog(
    rng: _Rng, snapshot: ScaffoldSnapshot, works: list[WorkSnapshot], family: str
) -> _ConvDraft:
    decoys = list(snapshot.decoy_themes)
    matching = [d for d in decoys if d.query_lang == family]
    decoy = rng.choice(matching or decoys)
    desc = decoy.theme
    abstain_result = {"results": [], "abstain": True}
    slots = {"plot_description": desc}
    abstain_turns = [
        *_tool_turns("search_by_plot", {"description": desc, "top_k": 10}, abstain_result),
        TrainingMessage(
            role="assistant", content=f"{_preamble('out_of_catalog', slots)}\n\n{ABSTAIN[family]}"
        ),
    ]
    mid_conv = rng.random() < 0.35
    titled = [w for w in works if _unambiguous_queries(w)]
    if mid_conv and titled:
        work = rng.choice(titled)
        query = rng.choice(_unambiguous_queries(work))
        rt = work.resolve_title[query]
        gv = work.get_versions["indian"]
        slots1 = {"title": query}
        turns = [
            TrainingMessage(
                role="user",
                content=rng.choice(USER_TEMPLATES["find_by_title"][family]).format(title=query),
            ),
            *_tool_turns("resolve_title", {"title": query}, rt),
            *_tool_turns("get_versions", {"work_id": work.work_id, "scope": "indian"}, gv),
            TrainingMessage(
                role="assistant",
                content=_versions_answer(
                    family, rt["candidates"][0]["matched_title"], gv, "find_by_title", slots1
                ),
            ),
            TrainingMessage(
                role="user",
                content=rng.choice(USER_TEMPLATES["out_of_catalog"][family]).format(desc=desc),
            ),
            *abstain_turns,
        ]
        return _ConvDraft(
            turns=turns,
            intent_labels=["find_by_title", "out_of_catalog"],
            slot_labels=[slots1, slots],
            entity_ids=_entity_ids(gv, work.work_id),
            plan={"behaviour": "out_of_catalog", "decoy": desc, "mid_conv": work.work_key},
        )
    turns = [
        TrainingMessage(
            role="user",
            content=rng.choice(USER_TEMPLATES["out_of_catalog"][family]).format(desc=desc),
        ),
        *abstain_turns,
    ]
    return _ConvDraft(
        turns=turns,
        intent_labels=["out_of_catalog"],
        slot_labels=[slots],
        entity_ids=[],
        plan={"behaviour": "out_of_catalog", "decoy": desc, "mid_conv": None},
    )


def generate(snapshot: ScaffoldSnapshot, config: ScaffoldConfig) -> list[TrainingConversation]:
    """The pure generator: (snapshot, config) -> scaffold-only conversations."""
    rng = _Rng(config.seed)
    behaviour_counts = _allocate(config.behaviour_shares, config.size)
    behaviours: list[str] = [b for b, n in sorted(behaviour_counts.items()) for _ in range(n)]
    lang_counts = _language_plan(config)
    langs: list[str] = [lang for lang, n in sorted(lang_counts.items()) for _ in range(n)]
    rng.shuffle(behaviours)
    rng.shuffle(langs)

    works = snapshot.works
    plot_works = [w for w in works if w.plot_excerpts and w.get_versions["indian"].get("versions")]
    titled_works = [w for w in works if _unambiguous_queries(w)]

    conversations: list[TrainingConversation] = []
    for i, (behaviour, query_lang) in enumerate(zip(behaviours, langs, strict=True)):
        if query_lang == "native":
            family = "native-ta" if rng.random() < 0.4 else "native-hi"
        else:
            family = query_lang
        if behaviour == "find_by_plot":
            draft = _gen_find_by_plot(rng, plot_works, works, family)
        elif behaviour in ("find_by_title", "list_versions"):
            draft = _gen_find_by_title(rng, titled_works, family, behaviour)
        elif behaviour == "refine":
            draft = _gen_refine(rng, works, family)
        elif behaviour == "disambiguate":
            draft = _gen_disambiguate(rng, snapshot, family)
        else:
            draft = _gen_out_of_catalog(rng, snapshot, titled_works, family)
        plan = {**draft.plan, "i": i, "lang": query_lang, "family": family, "seed": config.seed}
        conversations.append(
            TrainingConversation(
                conv_id=f"ft-{config.seed:04d}-{i:05d}",
                behaviour=behaviour,  # type: ignore[arg-type]
                query_lang=query_lang,
                turns=draft.turns,
                entity_ids=draft.entity_ids,
                intent_labels=draft.intent_labels,
                slot_labels=draft.slot_labels,
                scaffold_hash=hashlib.sha256(
                    json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                teacher=None,
            )
        )
    return conversations


def mix_stats(conversations: list[TrainingConversation]) -> dict[str, dict[str, int]]:
    """behaviour -> query_lang -> count (the DatasetCard.counts shape)."""
    stats: dict[str, dict[str, int]] = {}
    for conv in conversations:
        stats.setdefault(conv.behaviour, {})
        stats[conv.behaviour][conv.query_lang] = stats[conv.behaviour].get(conv.query_lang, 0) + 1
    return stats
