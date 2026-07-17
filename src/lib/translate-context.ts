/**
 * Translation context + glossary builder.
 *
 * The pipeline is two-phase:
 *
 *  Phase 1 — Research (run once per movie):
 *    Given a TMDB context bundle, ask DeepSeek to produce a "scene &
 *    character research brief" + a Sinhala glossary. This locks in
 *    consistent translations for proper nouns, slang, recurring
 *    phrases, register (formal/informal), and tone.
 *
 *  Phase 2 — Translation (run per subtitle batch):
 *    For each batch of N cues, send:
 *      - the locked glossary
 *      - the previous 3-5 translated cues (rolling context)
 *      - the current batch's English text
 *    DeepSeek returns a JSON array of Sinhala translations preserving
 *    cue order so we can zip them back onto the cues.
 *
 * Keeping the glossary server-side and reusing it across batches is
 * what gives the output the consistency that raw Google Translate
 * lacks.
 */

import type { TranslationContextBundle } from "@/lib/tmdb";
import { callDeepSeek, streamDeepSeek, DEFAULT_TRANSLATE_MODEL } from "@/lib/deepseek";
import type { SubtitleCue } from "@/lib/subtitle";
import { toonStringify } from "@/lib/toon";

export interface GlossaryEntry {
  english: string;
  sinhala: string;
  note?: string;
}

export interface ResearchBrief {
  summary: string;
  setting: string;
  tone: string;
  register: string;
  characters: { name: string; description: string; sinhala_name: string }[];
  locations: { name: string; sinhala_name: string }[];
  recurring_phrases: GlossaryEntry[];
  proper_nouns: GlossaryEntry[];
  cultural_notes: string;
  glossary: GlossaryEntry[];
}

const RESEARCH_SYSTEM_PROMPT = `You are a Sinhala subtitle translator and film/TV researcher.

You will be given structured metadata about a movie or TV show (title, plot, cast with character names, genres, keywords, etc.) and your job is to prepare a TRANSLATION BRIEF that a downstream translator agent will use to subtitle this title from English into Sinhala (Sinhala script: සිංහල අකුරු).

Your brief MUST:
  1. Summarise the plot in 4-6 sentences — focusing on the parts a translator needs to know to disambiguate words.
  2. Describe the setting (period, place, social class) because it heavily affects word choice.
  3. Identify the overall TONE (e.g. comedic, gritty, melodramatic, satirical) and REGISTER (formal, colloquial, slangy, period-archaic) — translator MUST match this.
  4. List every named character with a 1-2 sentence description AND a locked Sinhala transliteration of their name. Use phonetic transliteration that sounds natural to a Sinhala reader. Be consistent.
  5. List every named location with its Sinhala form.
  6. List recurring phrases / idioms / slogans / running gags the translator must keep consistent across the whole subtitle file.
  7. List proper nouns (ships, weapons, spells, organisations, fictional terms) with locked Sinhala forms.
  8. Add CULTURAL NOTES — anything a Sinhala-speaking viewer needs (e.g. untranslatable jokes, cultural equivalents, taboo words to soften, period accuracy).
  9. Produce a final GLOSSARY array consolidating everything a translator needs as {english, sinhala, note?} triples.

CRITICAL RULES:
  - Output MUST be valid JSON and nothing else.
  - All Sinhala strings MUST use Sinhala Unicode script (අ-෴).
  - Never mix Latin letters into Sinhala text except for proper nouns that are conventionally kept in English (e.g. "NASA", "FBI").
  - Transliterations must be the SAME every time the same name appears — pick once and commit.
  - Keep notes in English (for the downstream agent) but the actual sinhala_name / sinhala fields in Sinhala script.`;

export async function buildResearchBrief(
  ctx: TranslationContextBundle,
  apiKey: string,
  opts?: { signal?: AbortSignal }
): Promise<ResearchBrief> {
  const userPrompt = `Movie/TV metadata for translation brief:

${JSON.stringify(ctx, null, 2)}

Produce the translation brief JSON now. Schema:
{
  "summary": string,
  "setting": string,
  "tone": string,
  "register": string,
  "characters": [{"name": string, "description": string, "sinhala_name": string}],
  "locations": [{"name": string, "sinhala_name": string}],
  "recurring_phrases": [{"english": string, "sinhala": string, "note"?: string}],
  "proper_nouns": [{"english": string, "sinhala": string, "note"?: string}],
  "cultural_notes": string,
  "glossary": [{"english": string, "sinhala": string, "note"?: string}]
}`;

  const result = await callDeepSeek({
    apiKey,
    messages: [
      { role: "system", content: RESEARCH_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.2,
    responseFormat: "json_object",
    maxTokens: 3500,
    timeoutMs: 120_000,
    signal: opts?.signal,
  });

  let parsed: ResearchBrief;
  try {
    parsed = JSON.parse(result.content) as ResearchBrief;
  } catch (err) {
    throw new Error(
      `DeepSeek returned invalid JSON for research brief: ${result.content.slice(0, 200)}`
    );
  }
  return parsed;
}

/**
 * Stream the research brief as JSON. Yields raw text chunks as they
 * arrive from DeepSeek (for live display), and returns the parsed
 * ResearchBrief object when complete.
 *
 * Uses a SINGLE DeepSeek call with JSON mode — no separate
 * buildResearchBrief call needed. This avoids Netlify function
 * timeouts (the streaming call stays alive, but a second synchronous
 * call would exceed the 26s free-tier limit).
 */
export async function* streamResearchBriefJson(
  ctx: TranslationContextBundle,
  apiKey: string,
  opts?: { signal?: AbortSignal }
): AsyncGenerator<string, ResearchBrief, unknown> {
  const userPrompt = `Movie/TV metadata for translation brief:

${JSON.stringify(ctx, null, 2)}

Produce the translation brief JSON now. Use Sinhala Unicode script (අ-෴) for all sinhala/sinhala_name fields. Be specific and practical.`;

  let full = "";
  for await (const chunk of streamDeepSeek({
    apiKey,
    messages: [
      { role: "system", content: RESEARCH_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.2,
    timeoutMs: 200_000,
    signal: opts?.signal,
  })) {
    full += chunk;
    yield chunk;
  }

  // Parse the accumulated JSON.
  let parsed: ResearchBrief;
  try {
    // DeepSeek with json_object mode wraps in a top-level object.
    parsed = JSON.parse(full) as ResearchBrief;
  } catch {
    // Try to extract a JSON object from the text.
    const match = full.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        parsed = JSON.parse(match[0]) as ResearchBrief;
      } catch {
        throw new Error(
          "DeepSeek returned invalid JSON for research brief. Please try again."
        );
      }
    } else {
      throw new Error(
        "DeepSeek returned invalid JSON for research brief. Please try again."
      );
    }
  }
  return parsed;
}

/** Stream the brief as readable text — used for the live "research" panel. */
export async function* streamResearchBrief(
  ctx: TranslationContextBundle,
  apiKey: string,
  opts?: { signal?: AbortSignal }
): AsyncGenerator<string, ResearchBrief, unknown> {
  const userPrompt = `Movie/TV metadata for translation brief:

${JSON.stringify(ctx, null, 2)}

Produce the translation brief as readable Markdown with these sections:
  ## Summary
  ## Setting & Period
  ## Tone & Register
  ## Characters (name → sinhala transliteration, description)
  ## Locations
  ## Recurring Phrases
  ## Proper Nouns
  ## Cultural Notes
  ## Glossary (english | sinhala | note)

Use Sinhala Unicode script for all sinhala translations. Be specific and practical — this brief will be consumed by a translation agent.`;

  let full = "";
  for await (const chunk of streamDeepSeek({
    apiKey,
    messages: [
      { role: "system", content: RESEARCH_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.3,
    signal: opts?.signal,
  })) {
    full += chunk;
    yield chunk;
  }

  // Best-effort: not strictly needed for streaming endpoint, but useful.
  return {} as ResearchBrief;
}

const TRANSLATION_SYSTEM_PROMPT = `You are a professional English → Sinhala subtitle translator.

You are given a TOON (Token-Oriented Object Notation) payload containing:
  - brief: a locked TRANSLATION BRIEF (characters, glossary, tone, register, cultural notes)
  - previous: the last few already-translated cues (for flow consistency)
  - batch: the English cues to translate this turn

TOON grammar (for reading the input only — your OUTPUT is JSON):
  - "key: value"        inline scalar
  - "key:"              nested object/array starts on next indented line
  - "@"                 array item (object if next lines are indented, else inline scalar after @)
  - "[]" / "{}"         explicit empty array / empty object
  - "true" / "false"    booleans
  - "\\n" inside a value = literal newline
  - Lines starting with "#" are comments

Your job: return a JSON object { "translations": ["...", "...", ...] } where the i-th string is the Sinhala translation of the i-th cue in "batch", in EXACTLY the same order.

Rules:
  1. Use Sinhala Unicode script (අ-෴) for all Sinhala text.
  2. Honor the glossary: every glossary entry MUST use its locked sinhala form.
  3. Honor character name transliterations.
  4. Match the requested TONE and REGISTER exactly. Slang stays slang, formal stays formal.
  5. Keep subtitle readability: short, natural, conversational. One subtitle line ideally ≤ 42 chars; two lines max. Preserve the original line breaks of the cue (if the English cue has two lines, the Sinhala should too).
  6. Do NOT translate proper nouns already in the glossary's "sinhala" field — use that exact form.
  7. Do NOT add explanations, quotes, brackets, or notes inside translations.
  8. If a line is untranslatable (e.g. pure sound effect, music note), keep the original text unchanged.

ACCURACY — these matter more than brevity:
  9.  Translate MEANING, not words. Never produce a literal/word-for-word rendering if it would sound unnatural or change the meaning in Sinhala — prefer the closest natural Sinhala equivalent (idiom-for-idiom, not word-for-word) while staying faithful to what was actually said.
  10. Resolve ambiguity (pronouns, tense, formality, sarcasm, double meanings) using the "previous" cues and the "brief" context — do not guess a wrong sense just to translate fast.
  11. Preserve exact content: do not drop clauses, invent details, soften/strengthen meaning, or merge two distinct cues' meaning together, even under length pressure.
  12. Keep numbers, names, dates, and quoted terms factually exact.
  13. Do not change who is speaking or the grammatical subject/object of a sentence.
  14. If two readings are plausible, pick the one consistent with the surrounding "previous" cues and the brief's tone/setting — never the two ideas that are broadest/vaguest.
  15. Before writing each translation, silently check it against the glossary and against the previous cues for consistency; only the final Sinhala text goes in the output array — never show this checking process.
  16. Output JSON ONLY. No prose before/after.`;

export interface TranslateBatchInput {
  brief: ResearchBrief;
  previousCues: SubtitleCue[]; // already-translated rolling context
  currentCues: SubtitleCue[]; // untranslated, to translate
}

/** Per-attempt budget, well under any route's maxDuration so we can
 *  fail fast and retry with a smaller batch instead of letting
 *  Vercel's hard timeout kill the whole request. */
const TRANSLATE_ATTEMPT_TIMEOUT_MS = 25_000;

async function translateBatchOnce(
  input: TranslateBatchInput,
  apiKey: string,
  opts?: { signal?: AbortSignal }
): Promise<string[]> {
  const { brief, previousCues, currentCues } = input;

  // Build a single TOON payload containing the brief, previous cues,
  // and the batch to translate. This is ~30-50% smaller than the
  // equivalent JSON, saving DeepSeek input tokens on every call.
  //
  // We trim each cue down to just the fields the translator needs:
  //   idx, start, end, text (and `si` for previous cues).
  // We trim the brief down to just the fields the translator needs:
  //   summary, setting, tone, register, cultural_notes, characters,
  //   locations, glossary.
  const toonPayload = toonStringify({
    brief: {
      summary: brief.summary,
      setting: brief.setting,
      tone: brief.tone,
      register: brief.register,
      cultural_notes: brief.cultural_notes,
      characters: brief.characters.map((c) => ({
        name: c.name,
        sinhala: c.sinhala_name,
        note: c.description,
      })),
      locations: brief.locations.map((l) => ({
        name: l.name,
        sinhala: l.sinhala_name,
      })),
      glossary: brief.glossary.map((g) => ({
        en: g.english,
        si: g.sinhala,
        note: g.note,
      })),
    },
    previous: previousCues.map((c, i) => ({
      idx: i + 1,
      start: c.startRaw,
      end: c.endRaw,
      en: c.text,
      si: c.translated ?? "",
    })),
    batch: currentCues.map((c, i) => ({
      idx: i + 1,
      start: c.startRaw,
      end: c.endRaw,
      en: c.text,
    })),
  });

  const userPrompt = `Translate every cue in the "batch" array below into Sinhala, following the locked "brief" glossary exactly. Return JSON: { "translations": ["...", "...", ...] } in the same order as the batch.

# TOON PAYLOAD

${toonPayload}`;

  const result = await callDeepSeek({
    apiKey,
    model: DEFAULT_TRANSLATE_MODEL,
    messages: [
      { role: "system", content: TRANSLATION_SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    temperature: 0.15,
    responseFormat: "json_object",
    maxTokens: Math.min(8000, 400 + currentCues.length * 200),
    timeoutMs: TRANSLATE_ATTEMPT_TIMEOUT_MS,
    signal: opts?.signal,
  });

  let arr: string[];
  try {
    const parsed = JSON.parse(result.content) as { translations?: string[] };
    arr = parsed.translations ?? [];
  } catch {
    // Last-ditch: try to recover an array out of the raw text.
    const match = result.content.match(/\[\s*([\s\S]*?)\s*\]/);
    if (match) {
      try {
        arr = JSON.parse(`[${match[1]}]`);
      } catch {
        arr = [];
      }
    } else {
      arr = [];
    }
  }

  // Fallback: if the model returned fewer/more than expected, pad/truncate.
  while (arr.length < currentCues.length) arr.push("");
  if (arr.length > currentCues.length) arr = arr.slice(0, currentCues.length);

  return arr;
}

/**
 * Translate a batch of cues, automatically retrying with a smaller
 * (halved) batch if a call times out or errors — instead of the
 * whole request failing/timing out. Bottoms out at single-cue calls;
 * if even that fails, the cue is left untranslated (empty string)
 * rather than blocking the rest of the file.
 */
export async function translateBatch(
  input: TranslateBatchInput,
  apiKey: string,
  opts?: { signal?: AbortSignal; onProgress?: (done: number, total: number) => void }
): Promise<string[]> {
  const { brief, previousCues, currentCues } = input;

  try {
    const arr = await translateBatchOnce(input, apiKey, opts);
    opts?.onProgress?.(currentCues.length, currentCues.length);
    return arr;
  } catch (err) {
    if (opts?.signal?.aborted) throw err; // user cancelled — don't retry
    if (currentCues.length <= 1) {
      // Nothing smaller to fall back to — surface empty translation
      // rather than failing the whole file.
      opts?.onProgress?.(currentCues.length, currentCues.length);
      return currentCues.map(() => "");
    }
    const mid = Math.ceil(currentCues.length / 2);
    const firstHalf = currentCues.slice(0, mid);
    const secondHalf = currentCues.slice(mid);

    const firstResult = await translateBatch(
      { brief, previousCues, currentCues: firstHalf },
      apiKey,
      opts
    );
    // Thread the now-translated first half into the rolling context
    // for the second half so glossary/tone consistency isn't lost.
    const bridgedPrevious = [
      ...previousCues,
      ...firstHalf.map((c, i) => ({ ...c, translated: firstResult[i] })),
    ];
    const secondResult = await translateBatch(
      { brief, previousCues: bridgedPrevious, currentCues: secondHalf },
      apiKey,
      opts
    );
    return [...firstResult, ...secondResult];
  }
}

/**
 * Iterate over an entire subtitle file in batches, threading the
 * rolling context (last K translated cues) between calls so the
 * model can keep tone & terminology consistent across the whole file.
 */
export async function* translateAllInBatches(
  cues: SubtitleCue[],
  brief: ResearchBrief,
  apiKey: string,
  opts: {
    batchSize?: number;
    rollingContext?: number;
    signal?: AbortSignal;
    onCueTranslated?: (cueIndex: number, sinhala: string) => void;
  } = {}
): AsyncGenerator<{ done: number; total: number; cue: SubtitleCue }, void, unknown> {
  const batchSize = opts.batchSize ?? 6;
  const rolling = opts.rollingContext ?? 6;

  for (let i = 0; i < cues.length; i += batchSize) {
    if (opts.signal?.aborted) throw new Error("aborted");
    const batch = cues.slice(i, i + batchSize);
    const previousCues = cues.slice(Math.max(0, i - rolling), i);

    const translations = await translateBatch(
      { brief, previousCues, currentCues: batch },
      apiKey,
      { signal: opts.signal }
    );

    for (let j = 0; j < batch.length; j++) {
      batch[j].translated = translations[j];
      opts.onCueTranslated?.(i + j, translations[j]);
      yield {
        done: i + j + 1,
        total: cues.length,
        cue: batch[j],
      };
    }
  }
}
