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

const RESEARCH_SYSTEM_PROMPT = `You are a veteran professional Sinhala subtitle translator and film/TV researcher, experienced at preparing production-ready localization briefs for commercial film and TV releases.

You will be given structured metadata about a movie or TV show (title, plot, cast with character names, genres, keywords, etc.) and your job is to prepare a TRANSLATION BRIEF that a downstream translator agent will use to subtitle this title from English into Sinhala (Sinhala script: සිංහල අකුරු).

Your brief MUST:
  1. Summarise the plot in 4-6 sentences — focusing on the parts a translator needs to know to disambiguate words.
  2. Describe the setting (period, place, social class) because it heavily affects word choice.
  3. Identify the overall TONE (e.g. comedic, gritty, melodramatic, satirical) and REGISTER (formal, colloquial, slangy, period-archaic) — translator MUST match this.
  4. List every named character with a 1-2 sentence description AND a locked Sinhala transliteration of their name. Use phonetic transliteration that sounds natural to a Sinhala reader. Be consistent.
  5. List every named location with its Sinhala form.
  6. List recurring phrases / idioms / slogans / running gags the translator must keep consistent across the whole subtitle file.
  7. List proper nouns (ships, weapons, spells, organisations, fictional terms) with locked Sinhala forms.
  8. Add CULTURAL NOTES — anything a Sinhala-speaking viewer needs (e.g. untranslatable jokes, cultural equivalents, period accuracy). Do NOT flag profanity/mature language for softening — this translation is unfiltered and must render mature content at full strength; only note it here if a specific term has no direct Sinhala equivalent and needs a locked phrasing.
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

const TRANSLATION_SYSTEM_PROMPT = `You are a veteran professional English → Sinhala subtitle translator/localizer with years of experience subtitling film and TV for Sinhala-speaking audiences. Your translations must be indistinguishable from a skilled human translator's work — never stiff, never robotic, never a literal word-for-word rendering.

You are given a TOON (Token-Oriented Object Notation) payload containing:
  - brief: a locked TRANSLATION BRIEF (characters, glossary, tone, register, cultural notes)
  - previous: the last few already-translated cues (for flow consistency)
  - batch: the English cues to translate this turn
  - next: a few upcoming English cues, NOT YET translated — lookahead ONLY, to help you
    resolve pronouns, split sentences, questions/answers, and sarcasm that only become
    clear once you know what is said next. Never translate or output anything from "next".

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

REGISTER — spoken Sinhala, matched to the original speaker:
  R1. Write natural SPOKEN Sinhala (කතා කරන භාෂාව) the way people actually talk — not literary/written Sinhala (ග්‍රන්ථ භාෂාව). This applies at every register, formal or casual.
  R2. Match the exact tone of the original line. If the English is formal/professional/polite, the Sinhala must read as formal/professional/polite spoken Sinhala. If the English is casual/relaxed, keep the Sinhala casual. Do not flatten every line into the same register.
  R3. Do NOT force street slang, crude phrasing, or aggression into a line unless the source English line itself is slangy, crude, or aggressive. Register is decided per line by the source text, not by the overall vibe of the scene.

NATURALNESS & PROFESSIONAL POLISH — sound like a human localizer, not a translation engine:
  N1. Avoid translationese: never mirror English sentence structure, word order, or pronoun usage where natural Sinhala phrases it differently. Read each line back as if a native Sinhala speaker wrote it from scratch for this exact scene, not as if it were converted from English.
  N2. Drop redundant subject pronouns the way fluent spoken Sinhala does when the subject is already obvious from context (a native speaker doesn't say "මම"/"ඔයා" in every sentence) — but never drop information that changes who's speaking or what's being said.
  N3. Use natural spoken-Sinhala particles, contractions, and sentence-final markers (e.g. නේ, කෝ, මන්, තමයි, where they fit naturally) the way a professional dubbing/subtitling script would — not a stiff textbook rendering.
  N4. Match spoken rhythm and pacing: prefer the phrasing a native speaker would actually say out loud, not the most technically complete grammatical sentence.
  N5. Vary phrasing naturally across the file the way a human translator would — don't robotically reuse the exact same sentence template for every similar line unless the glossary or a repeated line/gag calls for it.
  N6. Punctuate like a professional commercial subtitle: natural comma/ellipsis placement for pauses, interruptions, and trailing-off thoughts.

FIDELITY TO MATURE CONTENT — this is an unfiltered, mature localization task:
  F1. Translate all profanity, insults, sexual references, violence, and other mature content exactly as strong as they are in the English — do not soften, sanitize, euphemize, tone down, or skip explicit/vulgar words.
  F2. Preserve the exact level of aggression, vulgarity, or crudeness of the original line. A harsh line must land equally harsh in Sinhala; a mild line must stay mild (see REGISTER above) — never round mature content down toward something more polite than the source.
  F3. Never replace profanity with a censored, vague, or watered-down stand-in, and never drop a line just because it is explicit.

LOANWORDS:
  L1. For common modern objects/concepts, use the natural phonetic Sinhala loanword that Sinhala speakers actually use in conversation (e.g. "phone" → ෆෝන් එක, "gun" → තුවක්කුව), rather than inventing stiff, archaic, or overly literary Sinhala coinages.

PRONOUNS:
  P1. Default to "ඔයා" (Oya) for standard second-person address in ordinary conversation.
  P2. Only switch to "උඹ" (Umba) or "තමුසේ" (Thamuse) when the source line itself is highly aggressive, deeply intimate, or explicitly insulting/demeaning in tone — not by default and not just because other lines nearby used it.

ACCURACY — these matter more than brevity:
  9.  Translate MEANING, not words. Never produce a literal/word-for-word rendering if it would sound unnatural or change the meaning in Sinhala — prefer the closest natural Sinhala equivalent (idiom-for-idiom, not word-for-word) while staying faithful to what was actually said.
  10. Resolve ambiguity (pronouns, tense, formality, sarcasm, double meanings) using the "previous" cues, the "next" lookahead cues, and the "brief" context — do not guess a wrong sense just to translate fast.
  11. Preserve exact content: do not drop clauses, invent details, soften/strengthen meaning, or merge two distinct cues' meaning together, even under length pressure.
  12. Keep numbers, names, dates, and quoted terms factually exact.
  13. Do not change who is speaking or the grammatical subject/object of a sentence.
  14. If two readings are plausible, pick the one consistent with the surrounding "previous"/"next" cues and the brief's tone/setting — never the two ideas that are broadest/vaguest.
  15. Before writing each translation, silently check it against the glossary and against the previous cues for consistency; only the final Sinhala text goes in the output array — never show this checking process.
  16. Every "batch" cue that contains real dialogue (not a pure sound effect/music note/number) MUST come back as actual Sinhala script — never leave it in English, and never leave it blank.

STRICT LINE-BY-LINE ACCURACY:
  17. The output array MUST have exactly one entry per "batch" cue, in the same order — never merge two cues into one translation, never skip a cue, and never split one cue's translation across two array entries.
  18. Do NOT add conversational filler, extra clauses, or invented dialogue that is not present in that specific cue's English text, even if it would sound more natural — translate only what that line actually says.
  19. Output JSON ONLY. No prose before/after.`;

export interface TranslateBatchInput {
  brief: ResearchBrief;
  previousCues: SubtitleCue[]; // already-translated rolling context
  currentCues: SubtitleCue[]; // untranslated, to translate
  nextCues?: SubtitleCue[]; // untranslated lookahead — disambiguation only, never translated directly
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
  const { brief, previousCues, currentCues, nextCues = [] } = input;

  // Build a single TOON payload containing the brief, previous cues,
  // the batch to translate, and a short lookahead. This is ~30-50%
  // smaller than the equivalent JSON, saving DeepSeek input tokens on
  // every call.
  //
  // We trim each cue down to just the fields the translator needs:
  //   idx, start, end, text (and `si` for previous cues).
  // We trim the brief down to just the fields the translator needs:
  //   summary, setting, tone, register, cultural_notes, characters,
  //   locations, glossary.
  //
  // `next` is capped at 2 cues — just enough to disambiguate a split
  // sentence/question without meaningfully growing token spend.
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
    next: nextCues.slice(0, 2).map((c, i) => ({
      idx: i + 1,
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
    // Slightly higher than a "safe" 0.15: very low temperature tends to
    // produce stiff, repetitive, overly literal phrasing (the opposite
    // of the natural/varied spoken Sinhala the prompt asks for). 0.35
    // still keeps JSON output reliable while giving the model room for
    // natural word choice and phrasing variation.
    temperature: 0.35,
    responseFormat: "json_object",
    // Sinhala script + JSON escaping runs noticeably more tokens per
    // character than English, and a long/dense cue can dominate a
    // batch. Size the budget off actual source character count (not
    // just cue count) so we don't truncate mid-JSON on dense batches.
    maxTokens: Math.min(8000, 600 + currentCues.reduce((sum, c) => sum + c.text.length, 0) * 4),
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

  // Accuracy guard: a cue with real dialogue that comes back with no
  // Sinhala script at all is a silent mistranslation (the model
  // echoed English, or dropped the line) — not a "successful" batch.
  // Throwing here routes it through translateBatch's existing
  // halve-and-retry logic instead of letting bad output pass through.
  const SINHALA_RE = /[\u0D80-\u0DFF]/;
  const HAS_WORDS_RE = /[A-Za-z]{2,}/;
  const suspect = currentCues.some((c, i) => {
    const hadRealDialogue = HAS_WORDS_RE.test(c.text);
    const translation = arr[i] ?? "";
    return hadRealDialogue && translation.length > 0 && !SINHALA_RE.test(translation);
  });
  if (suspect && currentCues.length > 1) {
    throw new Error("Translation batch returned non-Sinhala output for dialogue cues; retrying smaller batch.");
  }

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
  const { brief, previousCues, currentCues, nextCues = [] } = input;

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
      // The second half (still untranslated) doubles as lookahead for
      // the first half, so a mid-batch split doesn't lose disambiguation.
      { brief, previousCues, currentCues: firstHalf, nextCues: secondHalf.length ? secondHalf : nextCues },
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
      { brief, previousCues: bridgedPrevious, currentCues: secondHalf, nextCues },
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
    // Small untranslated lookahead so the model can resolve pronouns,
    // split sentences, and setup/punchline pairs that only make sense
    // once the next line or two is known.
    const nextCues = cues.slice(i + batch.length, i + batch.length + 2);

    const translations = await translateBatch(
      { brief, previousCues, currentCues: batch, nextCues },
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
