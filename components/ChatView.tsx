"use client";

import { useEffect, useRef } from "react";
import { FileText, KeyRound } from "lucide-react";

import { Composer } from "@/components/Composer";
import { Message, type ChatMessage } from "@/components/Message";
import type { DocumentInfo } from "@/lib/types";

const SUGGESTIONS = ["What is this document about?", "Summarize the key points", "What should I pay attention to?"];

type Props = {
  document: DocumentInfo;
  messages: ChatMessage[];
  busy: boolean;
  trialExhausted: boolean;
  onAsk: (question: string) => void;
  onAddKey: () => void;
};

export function ChatView({ document, messages, busy, trialExhausted, onAsk, onAddKey }: Props) {
  const bottom = useRef<HTMLDivElement>(null);
  const lastContent = messages.at(-1)?.content;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.length, lastContent]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-4 sm:px-8">
          <div className="flex items-center gap-3 rounded-2xl border border-line bg-surface p-4">
            <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-brand-soft text-brand">
              <FileText className="size-5" />
            </div>
            <div className="min-w-0">
              <p className="text-[0.7rem] font-bold tracking-[0.12em] text-muted uppercase">Currently chatting with</p>
              <p className="truncate font-display font-bold" title={document.name}>
                {document.name}
              </p>
            </div>
            <p className="ml-auto hidden shrink-0 text-sm text-muted sm:block">
              {document.pages} {document.pages === 1 ? "page" : "pages"} · {document.chunks} passages
            </p>
          </div>

          {document.pages < document.totalPages && (
            <p className="mt-3 text-sm text-muted">
              Trial mode indexed the first {document.pages} of {document.totalPages} pages. Add your own key and
              replace the document to index all of it.
            </p>
          )}

          {messages.length === 0 ? (
            <div className="py-12 text-center">
              <h2 className="font-display text-2xl font-bold tracking-tight">What would you like to know?</h2>
              <p className="mt-1 text-muted">Ask anything, or start with one of these.</p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    disabled={busy || trialExhausted}
                    onClick={() => onAsk(suggestion)}
                    className="rounded-full border border-line bg-surface px-4 py-2 text-sm transition hover:border-brand hover:text-brand disabled:opacity-50"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mt-6 space-y-5" aria-live="polite">
              {messages.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </div>
          )}
          <div ref={bottom} />
        </div>
      </div>

      <div className="border-t border-line bg-paper/90 backdrop-blur">
        <div className="mx-auto w-full max-w-3xl px-4 py-3 sm:px-8 sm:py-4">
          {trialExhausted && (
            <div className="mb-3 flex flex-col gap-3 rounded-xl bg-warn-soft p-3 text-sm text-warn sm:flex-row sm:items-center">
              <p className="flex-1">Your two-question trial is complete. Add your own OpenAI API key to keep asking.</p>
              <button
                type="button"
                onClick={onAddKey}
                className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-brand px-3 py-1.5 font-semibold text-white hover:bg-brand-hover dark:text-[#0c1310]"
              >
                <KeyRound className="size-4" /> Add key
              </button>
            </div>
          )}
          <Composer
            disabled={busy || trialExhausted}
            placeholder={trialExhausted ? "Add your API key to continue…" : "Ask a question about your document…"}
            onSend={onAsk}
          />
          <p className="mt-2 hidden text-center text-xs text-muted sm:block">
            Answers are generated from retrieved passages and may be incomplete. Check the sources.
          </p>
        </div>
      </div>
    </div>
  );
}
