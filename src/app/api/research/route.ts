import { NextRequest } from "next/server";
import { streamResearchBriefJson } from "@/lib/translate-context";
import type { TranslationContextBundle } from "@/lib/tmdb";
import { getCachedBrief, upsertCachedBrief } from "@/lib/brief-cache";
import { getCurrentUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

/**
 * POST /api/research
 *
 * Streams the research brief as raw text chunks (JSON from DeepSeek).
 *
 *   1. If a cached brief exists AND force_refresh is false → emit a
 *      "served from cache" notice, then [DONE]. No DeepSeek call.
 *
 *   2. Otherwise → stream the JSON brief live from DeepSeek. After
 *      streaming completes, parse the accumulated JSON and cache it.
 *      SINGLE DeepSeek call — no separate buildResearchBrief call
 *      (which would timeout on Netlify's free tier).
 */
export async function POST(req: NextRequest) {
  const user = await getCurrentUser();
  if (!user) {
    return new Response(JSON.stringify({ error: "Please log in to continue." }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });
  }

  const body = (await req.json().catch(() => ({}))) as {
    context?: TranslationContextBundle;
    tmdb_id?: number;
    tmdb_media_type?: "movie" | "tv";
    force_refresh?: boolean;
  };

  if (!body.context) {
    return new Response(JSON.stringify({ error: "Missing 'context'" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }
  if (!body.tmdb_id || !body.tmdb_media_type) {
    return new Response(
      JSON.stringify({ error: "Missing 'tmdb_id' or 'tmdb_media_type'" }),
      { status: 400, headers: { "content-type": "application/json" } }
    );
  }

  // 1. Try cache first.
  if (!body.force_refresh) {
    try {
      const cached = await getCachedBrief(body.tmdb_id, body.tmdb_media_type);
      if (cached) {
        const header =
          `[CACHE HIT] Loaded cached research brief for ${cached.title}.\n` +
          `Last updated: ${cached.updatedAt.toISOString()}\n` +
          `Click "Refresh" to re-run with AI.\n\n`;
        const stream = new ReadableStream({
          start(controller) {
            const enc = new TextEncoder();
            controller.enqueue(enc.encode(header));
            controller.enqueue(enc.encode(cached.rawMarkdown));
            controller.enqueue(enc.encode("\n\n[DONE]"));
            controller.close();
          },
        });
        return new Response(stream, {
          headers: {
            "content-type": "text/plain; charset=utf-8",
            "cache-control": "no-cache, no-transform",
            "x-accel-buffering": "no",
            "x-cache-hit": "true",
          },
        });
      }
    } catch (err) {
      console.error("[research] cache read failed:", err);
    }
  }

  // 2. Live research.
  const apiKey = process.env.DEEPSEEK_API_KEY || "";
  if (!apiKey) {
    return new Response(
      JSON.stringify({
        error: "Translation service is not configured. Please contact the site admin.",
      }),
      { status: 503, headers: { "content-type": "application/json" } }
    );
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // NOTE: do NOT add "\n" after each chunk — DeepSeek streams
      // Sinhala text one character at a time, and adding "\n" would
      // break every character onto its own line. Just send raw chunks.
      const send = (s: string) => controller.enqueue(encoder.encode(s));
      try {
        send("[LIVE] Researching the movie with AI...\n\n");
        let full = "";
        const gen = streamResearchBriefJson(body.context!, apiKey);
        let brief;
        while (true) {
          const result = await gen.next();
          if (result.done) {
            brief = result.value;
            break;
          }
          full += result.value;
          send(result.value);
        }

        // Cache the parsed brief.
        if (brief) {
          try {
            await upsertCachedBrief({
              tmdbId: body.tmdb_id!,
              tmdbMediaType: body.tmdb_media_type!,
              title: body.context!.title,
              rawMarkdown: full,
              brief,
            });
            send("\n\n[INFO] Brief cached. You can now translate subtitles.");
          } catch (cacheErr: any) {
            send(`\n\n[WARN] Failed to cache brief: ${cacheErr.message}`);
          }
        } else {
          send("\n\n[WARN] No structured brief was returned.");
        }

        send("\n\n[DONE]");
      } catch (err: any) {
        const msg = err.message || "";
        let friendly = msg;
        if (msg.includes("401") || msg.includes("Authentication Fails")) {
          friendly = "Translation service authentication failed. Please contact the site admin.";
        } else if (msg.includes("429")) {
          friendly = "Translation service is busy. Please try again in a moment.";
        } else if (msg.includes("timed out") || msg.includes("timeout")) {
          friendly = "Research timed out. Try again — research can take 30-60 seconds for complex movies.";
        }
        send(`\n\n[ERROR] ${friendly}`);
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
      "x-cache-hit": "false",
    },
  });
}
