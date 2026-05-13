"use client";

import React, { useCallback, useRef, useState } from "react";
import { AlignEnd } from "./AlignEnd";
import { Button } from "./Button";
import { InputContainer } from "./Container";
import { ErrorComp } from "./Error";
import { Spacing } from "./Spacing";

type Status = "idle" | "open" | "generating" | "done" | "error";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "https://849ams2zdun0ya-8000.proxy.runpod.net";

export const WanGenerator: React.FC = () => {
  const [status, setStatus]   = useState<Status>("idle");
  const [prompt, setPrompt]   = useState("");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const prevUrlRef            = useRef<string | null>(null);

  const generate = useCallback(async () => {
    if (!prompt.trim()) return;
    setStatus("generating");
    setError(null);

    // Révoque l'ancienne URL objet pour libérer la mémoire
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current);
      prevUrlRef.current = null;
    }

    try {
      const res = await fetch(`${API_URL}/wan/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Erreur ${res.status}: ${text}`);
      }

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      prevUrlRef.current = url;
      setVideoUrl(url);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [prompt]);

  const reset = useCallback(() => {
    setStatus("open");
    setVideoUrl(null);
    setError(null);
  }, []);

  // Bouton d'entrée quand le panneau est fermé
  if (status === "idle") {
    return (
      <Button secondary onClick={() => setStatus("open")}>
        Générer avec Wan 2.2
      </Button>
    );
  }

  return (
    <InputContainer>
      <label className="text-sm font-medium text-foreground">
        Wan 2.2 — Texte vers vidéo
      </label>
      <Spacing />

      {/* Champ prompt */}
      {status === "open" || status === "generating" || status === "error" ? (
        <textarea
          className="leading-[1.7] block w-full rounded-geist bg-background p-geist-half text-foreground text-sm border border-unfocused-border-color transition-colors duration-150 ease-in-out focus:border-focused-border-color outline-none resize-none"
          rows={3}
          placeholder="Décris la vidéo à générer…"
          disabled={status === "generating"}
          value={prompt}
          onChange={(e) => setPrompt(e.currentTarget.value)}
        />
      ) : null}

      {status === "error" && error ? (
        <ErrorComp message={error} />
      ) : null}

      <Spacing />

      {/* Vidéo générée */}
      {status === "done" && videoUrl ? (
        <>
          <video
            src={videoUrl}
            controls
            autoPlay
            loop
            className="w-full rounded-geist overflow-hidden"
          />
          <Spacing />
          <a
            href={videoUrl}
            download="wan-output.mp4"
            className="text-xs text-foreground underline self-start"
          >
            Télécharger
          </a>
          <Spacing />
        </>
      ) : null}

      {/* Boutons d'action */}
      <div className="flex gap-2 justify-between items-center">
        <button
          className="text-xs text-foreground opacity-50 hover:opacity-100 transition-opacity"
          onClick={() => setStatus("idle")}
        >
          Fermer
        </button>
        <AlignEnd>
          {status === "done" ? (
            <Button secondary onClick={reset}>
              Nouvelle vidéo
            </Button>
          ) : (
            <Button
              disabled={status === "generating" || !prompt.trim()}
              loading={status === "generating"}
              onClick={generate}
            >
              {status === "generating" ? "Génération…" : "Générer"}
            </Button>
          )}
        </AlignEnd>
      </div>
    </InputContainer>
  );
};
