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

/** Default model for research/one-off calls. Override with DEEPSEEK_MODEL. */
export const DEFAULT_DEEPSEEK_MODEL =
  process.env.DEEPSEEK_MODEL?.trim() || "deepseek-v4-pro";

/**
 * Default model for high-volume translation batches, where latency
 * compounds across many calls. Override with DEEPSEEK_TRANSLATE_MODEL.
 */
export const DEFAULT_TRANSLATE_MODEL =
  process.env.DEEPSEEK_TRANSLATE_MODEL?.trim() || "deepseek-v4-flash";

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
  if (!opts.apiKey) {
    throw new Error(
      "DeepSeek API key is missing. Set DEEPSEEK_API_KEY on the server or pass it via the UI settings."
    );
  }

  const body: Record<string, unknown> = {
    model: opts.model ?? DEFAULT_DEEPSEEK_MODEL,
    messages: opts.messages,
    stream: false,
    thinking: { type: opts.thinking ? "enabled" : "disabled" },
  };
  // Thinking mode ignores temperature — only send it in non-thinking calls.
  if (!opts.thinking) body.temperature = opts.temperature ?? 0.2;
  if (opts.maxTokens) body.max_tokens = opts.maxTokens;
  if (opts.responseFormat === "json_object") {
    body.response_format = { type: "json_object" };
  }

  const { signal, cleanup } = withTimeout(opts.timeoutMs, opts.signal);
  let res: Response;
  try {
    res = await fetch(`${DEEPSEEK_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${opts.apiKey}`,
        "Content-Type": "application/json",
      },
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
  if (!opts.apiKey) {
    throw new Error("DeepSeek API key is missing.");
  }
  const body: Record<string, unknown> = {
    model: opts.model ?? DEFAULT_DEEPSEEK_MODEL,
    messages: opts.messages,
    stream: true,
    thinking: { type: opts.thinking ? "enabled" : "disabled" },
  };
  if (!opts.thinking) body.temperature = opts.temperature ?? 0.3;
  if (opts.maxTokens) body.max_tokens = opts.maxTokens;
  if (opts.responseFormat === "json_object") {
    body.response_format = { type: "json_object" };
  }

  const { signal, cleanup } = withTimeout(opts.timeoutMs, opts.signal);
  let res: Response;
  try {
    res = await fetch(`${DEEPSEEK_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${opts.apiKey}`,
        "Content-Type": "application/json",
      },
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
