/**
 * DeepSeek API client.
 *
 * DeepSeek exposes an OpenAI-compatible /chat/completions endpoint.
 * We target the DeepSeek V4 model family:
 *
 *   - deepseek-v4-pro   — 1.6T/49B MoE, strongest reasoning & world
 *                          knowledge but noticeably slower. Used for
 *                          the one-off, quality-critical research
 *                          brief (built once per movie).
 *   - deepseek-v4-flash — 284B/13B MoE, much faster and cheaper,
 *                          still strong. Used as the default for
 *                          translation batches, which run many times
 *                          per subtitle file — this is what was
 *                          timing out under v4-pro's latency. Overall
 *                          accuracy comes mainly from the locked
 *                          glossary/brief (v4-pro) + strict prompting,
 *                          not from which model translates each
 *                          batch, so flash keeps quality high while
 *                          fixing the timeouts.
 *
 * The legacy `deepseek-chat` / `deepseek-reasoner` aliases are being
 * retired by DeepSeek on 2026-07-24, so we no longer reference them.
 *
 * V4 defaults to "thinking mode" (extended reasoning, returned in a
 * separate `reasoning_content` field). We explicitly request
 * `thinking: disabled` unless the caller opts in, because:
 *   1. Our calls use JSON mode (`response_format: json_object`) for
 *      machine-parsed output, and thinking + JSON mode together is
 *      known to be flaky on some DeepSeek deployments.
 *   2. Thinking mode ignores `temperature` and burns significantly
 *      more output tokens/latency — risky for serverless function
 *      time limits (e.g. on Vercel) when translating many subtitle
 *      batches back-to-back.
 * Callers that want deeper reasoning (e.g. a slower, higher-quality
 * one-off research pass) can pass `thinking: true` explicitly.
 *
 * The API key MUST be passed server-side only — never expose it to
 * the browser. The UI collects it from the user, but it is sent
 * only inside server-side fetch calls.
 */

const DEEPSEEK_BASE = "https://api.deepseek.com/v1";
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

// OpenRouter config — must be evaluated before DEFAULT_DEEPSEEK_MODEL /
// DEFAULT_TRANSLATE_MODEL, because those exported constants are
// imported by callers (translate-context.ts, ai-search/route.ts) and
// passed explicitly as `model:` in the request body. They therefore
// need to resolve to the OpenRouter model id when OpenRouter is active.
//
// IMPORTANT: OpenRouter requires the FULLY-QUALIFIED model id
// (provider/model-variant). The bare name "gemma-4-26b-a4b" is NOT
// a valid OpenRouter id — OpenRouter returns 404 for it, which the
// translateBatch halve-and-retry logic silently swallows, producing
// empty translations and the "shows translating but doesn't update"
// bug. The correct id confirmed via `GET /api/v1/models` is:
//   google/gemma-4-26b-a4b-it
// (Use google/gemma-4-26b-a4b-it:free for the free-tier variant.)
//
// If a user explicitly sets OPENROUTER_MODEL we honor their value
// verbatim — that lets them target the :free variant, gemma-4-31b-it,
// or any other OpenRouter model without code changes.
const OPENROUTER_DEFAULT_MODEL = "google/gemma-4-26b-a4b-it";

function isOpenRouterEnabled(): boolean {
  return !!process.env.OPENROUTER_API_KEY?.trim();
}

/**
 * Default model for research/one-off calls.
 *
 *   DeepSeek:    deepseek-v4-pro   (override via DEEPSEEK_MODEL)
 *   OpenRouter:  gemma-4-26b-a4b  (override via OPENROUTER_MODEL)
 */
export const DEFAULT_DEEPSEEK_MODEL = isOpenRouterEnabled()
  ? (process.env.OPENROUTER_MODEL?.trim() || OPENROUTER_DEFAULT_MODEL)
  : (process.env.DEEPSEEK_MODEL?.trim() || "deepseek-v4-pro");

/**
 * Default model for high-volume translation batches, where latency
 * compounds across many calls.
 *
 *   DeepSeek:    deepseek-v4-flash  (override via DEEPSEEK_TRANSLATE_MODEL)
 *   OpenRouter:  defaults to OPENROUTER_MODEL (or gemma-4-26b-a4b).
 *                Override via OPENROUTER_TRANSLATE_MODEL.
 */
export const DEFAULT_TRANSLATE_MODEL = isOpenRouterEnabled()
  ? (process.env.OPENROUTER_TRANSLATE_MODEL?.trim() ||
      process.env.OPENROUTER_MODEL?.trim() ||
      OPENROUTER_DEFAULT_MODEL)
  : (process.env.DEEPSEEK_TRANSLATE_MODEL?.trim() || "deepseek-v4-flash");

/**
 * OpenRouter support.
 *
 * OpenRouter exposes the same OpenAI-compatible /chat/completions
 * endpoint as DeepSeek, so the same client can target either
 * provider. We switch based on env vars — no caller changes:
 *
 *   OPENROUTER_API_KEY         — if set, OpenRouter is used INSTEAD
 *                                of DeepSeek for every call.
 *   OPENROUTER_MODEL           — model id for research/one-off calls.
 *                                Defaults to "google/gemma-4-26b-a4b-it"
 *                                (the fully-qualified id that OpenRouter
 *                                requires — the bare "gemma-4-26b-a4b"
 *                                is NOT valid and will 404).
 *   OPENROUTER_TRANSLATE_MODEL — model id for high-volume translation
 *                                batches. Defaults to OPENROUTER_MODEL.
 *   OPENROUTER_REFERER         — optional, sent as HTTP-Referer header
 *                                for OpenRouter attribution.
 *   OPENROUTER_APP_TITLE       — optional, sent as X-Title header.
 *
 * When OpenRouter is active, the DeepSeek-only `thinking` request
 * field is omitted. The OpenAI-standard fields (model, messages,
 * temperature, max_tokens, response_format, stream) are sent exactly
 * as before — OpenRouter's gemma-4-26b-a4b-it supports `response_format`
 * natively per its supported_parameters list.
 */
interface ProviderConfig {
  baseUrl: string;
  apiKey: string;
  /** Whether to send the DeepSeek V4 `thinking` field in the body. */
  supportsThinking: boolean;
  /**
   * Whether to send `response_format: { type: "json_object" }` when
   * the caller asks for JSON output. Both DeepSeek and OpenRouter's
   * Gemma 4 model list `response_format` in their supported parameters,
   * so this is true for both providers. Kept as a flag in case a
   * future OpenRouter model rejects it.
   */
  supportsJsonResponseFormat: boolean;
  /** Extra HTTP headers (e.g. OpenRouter attribution headers). */
  extraHeaders?: Record<string, string>;
}

/**
 * Resolve which LLM provider + base URL + key to use for this call.
 *
 * The caller still passes a DeepSeek-style `apiKey`, but if OpenRouter
 * is configured we OVERRIDE it with the OpenRouter key from env. This
 * keeps the existing call sites (which all pass `process.env.DEEPSEEK_API_KEY`)
 * working unchanged — they don't need to know which provider is active.
 *
 * The model id is resolved by the caller (via DEFAULT_DEEPSEEK_MODEL /
 * DEFAULT_TRANSLATE_MODEL, which themselves respect the OpenRouter env).
 */
function resolveProvider(callerApiKey: string): ProviderConfig {
  if (isOpenRouterEnabled()) {
    return {
      baseUrl: OPENROUTER_BASE,
      apiKey: process.env.OPENROUTER_API_KEY!.trim(),
      supportsThinking: false,
      // OpenRouter's metadata (GET /api/v1/models) confirms that
      // google/gemma-4-26b-a4b-it DOES list `response_format` and
      // `structured_outputs` in supported_parameters. Sending
      // `response_format: { type: "json_object" }` is therefore safe
      // and makes the JSON output reliable (no regex fallback needed).
      supportsJsonResponseFormat: true,
      extraHeaders: {
        "HTTP-Referer":
          process.env.OPENROUTER_REFERER?.trim() ||
          "https://subsinhala.app",
        "X-Title":
          process.env.OPENROUTER_APP_TITLE?.trim() || "SubSinhala",
      },
    };
  }
  return {
    baseUrl: DEEPSEEK_BASE,
    apiKey: callerApiKey,
    supportsThinking: true,
    supportsJsonResponseFormat: true,
  };
}

export interface DeepSeekMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface DeepSeekCallOptions {
  apiKey: string;
  model?: string;
  messages: DeepSeekMessage[];
  temperature?: number;
  maxTokens?: number;
  responseFormat?: "text" | "json_object";
  /**
   * Enable DeepSeek V4 "thinking mode" (extended chain-of-thought,
   * returned separately in `reasoning_content`). Defaults to false —
   * see module docblock for why.
   */
  thinking?: boolean;
  /**
   * Abort the request after this many ms and throw a `DeepSeekTimeoutError`.
   * Lets callers fail fast and retry (e.g. with a smaller batch) well
   * before the serverless function's own hard maxDuration kills the
   * whole request with an opaque 504.
   */
  timeoutMs?: number;
  signal?: AbortSignal;
}

export class DeepSeekTimeoutError extends Error {
  constructor(ms: number) {
    super(`DeepSeek call timed out after ${ms}ms`);
    this.name = "DeepSeekTimeoutError";
  }
}

/** Combine a caller-supplied AbortSignal with an internal timeout. */
function withTimeout(
  timeoutMs: number | undefined,
  outerSignal: AbortSignal | undefined
): { signal: AbortSignal | undefined; cleanup: () => void } {
  if (!timeoutMs) return { signal: outerSignal, cleanup: () => {} };
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new DeepSeekTimeoutError(timeoutMs)),
    timeoutMs
  );
  const onOuterAbort = () => controller.abort(outerSignal?.reason);
  outerSignal?.addEventListener("abort", onOuterAbort);
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      outerSignal?.removeEventListener("abort", onOuterAbort);
    },
  };
}

export interface DeepSeekCallResult {
  content: string;
  finish_reason: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export async function callDeepSeek(
  opts: DeepSeekCallOptions
): Promise<DeepSeekCallResult> {
  const provider = resolveProvider(opts.apiKey);
  if (!provider.apiKey) {
    throw new Error(
      "LLM API key is missing. Set DEEPSEEK_API_KEY (or OPENROUTER_API_KEY to use OpenRouter with gemma-4-26b-a4b) on the server."
    );
  }

  const body: Record<string, unknown> = {
    model: opts.model ?? DEFAULT_DEEPSEEK_MODEL,
    messages: opts.messages,
    stream: false,
  };
  if (provider.supportsThinking) {
    body.thinking = { type: opts.thinking ? "enabled" : "disabled" };
  }
  // Thinking mode ignores temperature — only send it in non-thinking calls.
  if (!opts.thinking) body.temperature = opts.temperature ?? 0.2;
  if (opts.maxTokens) body.max_tokens = opts.maxTokens;
  if (
    provider.supportsJsonResponseFormat &&
    opts.responseFormat === "json_object"
  ) {
    body.response_format = { type: "json_object" };
  }

  const { signal, cleanup } = withTimeout(opts.timeoutMs, opts.signal);
  let res: Response;
  try {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${provider.apiKey}`,
      "Content-Type": "application/json",
    };
    if (provider.extraHeaders) {
      Object.assign(headers, provider.extraHeaders);
    }
    res = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (signal?.aborted && signal.reason instanceof DeepSeekTimeoutError) {
      throw signal.reason;
    }
    throw err;
  } finally {
    cleanup();
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`DeepSeek ${res.status}: ${text}`);
  }

  const data = await res.json();
  const choice = data.choices?.[0];
  if (!choice) {
    throw new Error("DeepSeek returned no choices");
  }
  return {
    content: choice.message?.content ?? "",
    finish_reason: choice.finish_reason,
    usage: data.usage ?? {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

/**
 * Streamed DeepSeek call. Yields incremental text chunks as they
 * arrive — used for the "research" panel so the user can watch the
 * model think through the movie context in real time.
 */
export async function* streamDeepSeek(
  opts: DeepSeekCallOptions
): AsyncGenerator<string, void, unknown> {
  const provider = resolveProvider(opts.apiKey);
  if (!provider.apiKey) {
    throw new Error(
      "LLM API key is missing. Set DEEPSEEK_API_KEY (or OPENROUTER_API_KEY to use OpenRouter with gemma-4-26b-a4b) on the server."
    );
  }
  const body: Record<string, unknown> = {
    model: opts.model ?? DEFAULT_DEEPSEEK_MODEL,
    messages: opts.messages,
    stream: true,
  };
  if (provider.supportsThinking) {
    body.thinking = { type: opts.thinking ? "enabled" : "disabled" };
  }
  if (!opts.thinking) body.temperature = opts.temperature ?? 0.3;
  if (opts.maxTokens) body.max_tokens = opts.maxTokens;
  if (
    provider.supportsJsonResponseFormat &&
    opts.responseFormat === "json_object"
  ) {
    body.response_format = { type: "json_object" };
  }

  const { signal, cleanup } = withTimeout(opts.timeoutMs, opts.signal);
  let res: Response;
  try {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${provider.apiKey}`,
      "Content-Type": "application/json",
    };
    if (provider.extraHeaders) {
      Object.assign(headers, provider.extraHeaders);
    }
    res = await fetch(`${provider.baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    cleanup();
    if (signal?.aborted && signal.reason instanceof DeepSeekTimeoutError) {
      throw signal.reason;
    }
    throw err;
  }

  if (!res.ok || !res.body) {
    cleanup();
    const text = await res.text().catch(() => "");
    throw new Error(`DeepSeek ${res.status}: ${text}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta.length > 0) {
            yield delta;
          }
        } catch {
          // ignore partial JSON
        }
      }
    }
  } catch (err) {
    if (signal?.aborted && signal.reason instanceof DeepSeekTimeoutError) {
      throw signal.reason;
    }
    throw err;
  } finally {
    cleanup();
  }
}
