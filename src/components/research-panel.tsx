"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sparkles,
  Loader2,
  AlertCircle,
  RefreshCw,
  Database,
} from "lucide-react";
import type { TranslationContextBundle } from "@/lib/tmdb";
import type { ResearchBrief } from "@/lib/translate-context";

interface ResearchPanelProps {
  context: TranslationContextBundle | null;
  tmdbId: number | null;
  tmdbMediaType: "movie" | "tv" | null;
  onBriefReady: (brief: ResearchBrief) => void;
  onBriefVersionChange?: (version: number) => void;
}

export function ResearchPanel({
  context,
  tmdbId,
  tmdbMediaType,
  onBriefReady,
  onBriefVersionChange,
}: ResearchPanelProps) {
  const [streaming, setStreaming] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cacheHit, setCacheHit] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const { toast } = useToast();

  const loadFromCache = useCallback(async () => {
    if (!tmdbId || !tmdbMediaType) {
      setText("");
      setCacheHit(false);
      return;
    }
    try {
      const res = await fetch(
        `/api/brief/get?tmdb_id=${tmdbId}&tmdb_media_type=${tmdbMediaType}`
      );
      if (res.status === 404) {
        setText("");
        setCacheHit(false);
        return;
      }
      const data = await res.json();
      if (data.cached && data.brief) {
        const header =
          `[CACHE HIT] Loaded cached research brief for ${data.title}.\n` +
          `Last updated: ${new Date(data.updatedAt).toLocaleString()}\n` +
          `Click "Refresh" to re-run with AI.\n\n`;
        setText(header + "(Cached — open the Glossary tab to view locked terms.)");
        setCacheHit(true);
        onBriefReady(data.brief);
        onBriefVersionChange?.(Date.now());
      }
    } catch (err) {
      console.error("Failed to load cached brief:", err);
    }
  }, [tmdbId, tmdbMediaType, onBriefReady, onBriefVersionChange]);

  useEffect(() => {
    loadFromCache();
  }, [loadFromCache]);

  const run = useCallback(
    async (forceRefresh: boolean = false) => {
      if (!context) return;
      if (!tmdbId || !tmdbMediaType) {
        toast({
          title: "Pick a movie first",
          variant: "destructive",
        });
        return;
      }
      setStreaming(true);
      setText("");
      setError(null);
      setCacheHit(false);
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      try {
        const res = await fetch("/api/research", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            context,
            tmdb_id: tmdbId,
            tmdb_media_type: tmdbMediaType,
            force_refresh: forceRefresh,
          }),
          signal: ac.signal,
        });

        if (!res.ok || !res.body) {
          let errMsg = `Request failed: ${res.status}`;
          try {
            const errJson = await res.json();
            errMsg = errJson.error || errMsg;
          } catch {
            const errText = await res.text();
            if (errText) errMsg = errText;
          }
          throw new Error(errMsg);
        }

        const wasCacheHit = res.headers.get("x-cache-hit") === "true";
        setCacheHit(wasCacheHit);

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let full = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          full += chunk;
          setText(full);
        }

        if (full.includes("[ERROR]")) {
          throw new Error(full.split("[ERROR]")[1]?.trim() || "Research failed");
        }

        // Fetch the structured brief that was just cached.
        const briefRes = await fetch(
          `/api/brief/get?tmdb_id=${tmdbId}&tmdb_media_type=${tmdbMediaType}`
        );
        if (briefRes.ok) {
          const briefData = await briefRes.json();
          if (briefData.brief) {
            onBriefReady(briefData.brief);
            onBriefVersionChange?.(Date.now());
          }
        }

        toast({
          title: wasCacheHit ? "Loaded from cache" : "Research complete",
          description: wasCacheHit
            ? "No AI call needed."
            : "Translation context is ready.",
        });
      } catch (err: any) {
        if (err.name !== "AbortError") {
          setError(err.message);
          toast({
            title: "Research failed",
            description: err.message,
            variant: "destructive",
          });
        }
      } finally {
        setStreaming(false);
      }
    },
    [context, tmdbId, tmdbMediaType, onBriefReady, onBriefVersionChange, toast]
  );

  function stop() {
    abortRef.current?.abort();
    setStreaming(false);
  }

  return (
    <Card className="flex flex-col p-4 gap-3 h-full">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Sparkles className="h-4 w-4 text-primary" />
          <h3 className="font-semibold">Research</h3>
          {streaming && (
            <Badge variant="secondary" className="gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Researching
            </Badge>
          )}
          {!streaming && cacheHit && (
            <Badge variant="outline" className="gap-1">
              <Database className="h-3 w-3" />
              Cached
            </Badge>
          )}
          {!streaming && text && !cacheHit && (
            <Badge variant="outline">Ready</Badge>
          )}
        </div>
        <div className="flex items-center gap-1">
          {!streaming && cacheHit && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => run(true)}
              disabled={!context || streaming}
              className="gap-1"
              title="Re-run research and overwrite the cached brief"
            >
              <RefreshCw className="h-3 w-3" />
              Refresh
            </Button>
          )}
          {!streaming ? (
            <Button
              size="sm"
              onClick={() => run(false)}
              disabled={!context || streaming}
              className="gap-1"
            >
              <Sparkles className="h-3 w-3" />
              {text ? "Reload" : "Run Research"}
            </Button>
          ) : (
            <Button size="sm" variant="outline" onClick={stop}>
              Stop
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <ScrollArea className="flex-1 min-h-0 rounded-md border bg-muted/30">
        <div className="p-3">
          {text ? (
            <pre className="whitespace-pre-wrap sinhala-serif text-sm leading-relaxed" lang="si">
              {text}
            </pre>
          ) : (
            <div className="text-sm text-muted-foreground italic">
              {context
                ? "Click \"Run Research\" to analyse this movie."
                : "Pick a movie first to enable research."}
            </div>
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}
