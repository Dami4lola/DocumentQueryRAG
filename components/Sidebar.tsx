"use client";

import { useState } from "react";
import { AlertTriangle, Eye, EyeOff, FileText, KeyRound, RefreshCw, Trash2, Upload, X } from "lucide-react";

import type { ConfigResponse, DocumentInfo } from "@/lib/types";

type Props = {
  config: ConfigResponse | null;
  userKey: string;
  onUserKeyChange: (key: string) => void;
  document: DocumentInfo | null;
  busy: boolean;
  questionsUsed: number;
  onPickFile: () => void;
  onRemoveDocument: () => void;
  onClose: () => void;
};

export function Sidebar({
  config,
  userKey,
  onUserKeyChange,
  document,
  busy,
  questionsUsed,
  onPickFile,
  onRemoveDocument,
  onClose,
}: Props) {
  const [showKey, setShowKey] = useState(false);
  const unlimited = Boolean(userKey.trim());
  const trialQuestions = config?.trialQuestions ?? 2;
  const trialPages = config?.trialPages ?? 5;
  const remaining = Math.max(0, trialQuestions - questionsUsed);

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto px-5 py-6">
      <div className="flex items-center justify-between">
        <div className="font-display text-xl font-extrabold tracking-tight text-brand">
          papertrail<span className="text-muted/60">.</span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1.5 text-muted hover:bg-line/60 lg:hidden"
          aria-label="Close sidebar"
        >
          <X className="size-5" />
        </button>
      </div>

      {config && config.missing.some((name) => name.startsWith("PINECONE")) && (
        <Notice tone="danger">
          The server is missing{" "}
          {config.missing.filter((name) => name.startsWith("PINECONE")).map((name, i) => (
            <span key={name}>
              {i > 0 && " and "}
              <code className="font-mono text-xs">{name}</code>
            </span>
          ))}
          . Documents can&apos;t be indexed until it&apos;s set.
        </Notice>
      )}

      <section>
        <SectionLabel>Connect</SectionLabel>
        <label htmlFor="openai-key" className="mb-1.5 flex items-center gap-1.5 text-sm font-medium">
          <KeyRound className="size-4 text-muted" /> OpenAI API key
        </label>
        <div className="relative">
          <input
            id="openai-key"
            type={showKey ? "text" : "password"}
            value={userKey}
            onChange={(event) => onUserKeyChange(event.target.value)}
            placeholder="sk-… for unlimited questions"
            autoComplete="off"
            spellCheck={false}
            className="w-full rounded-xl border border-line bg-surface py-2.5 pr-10 pl-3 text-sm outline-none placeholder:text-muted/70 focus:border-brand focus:ring-2 focus:ring-brand/20"
          />
          <button
            type="button"
            onClick={() => setShowKey((value) => !value)}
            className="absolute inset-y-0 right-0 grid w-10 place-items-center text-muted hover:text-ink"
            aria-label={showKey ? "Hide key" : "Show key"}
          >
            {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          Used only for your requests and kept in this browser tab. Never stored on the server.
        </p>

        <div className="mt-3 rounded-xl border border-line bg-surface p-3">
          {unlimited ? (
            <div className="flex items-center gap-2 text-sm">
              <span className="size-2 rounded-full bg-brand" />
              <span className="font-medium">Unlimited</span>
              <span className="text-muted">· your key</span>
            </div>
          ) : config && !config.hasOpenAIKey ? (
            <div className="flex items-center gap-2 text-sm">
              <span className="size-2 rounded-full bg-warn" />
              <span className="font-medium">Key required</span>
              <span className="text-muted">· no demo key set</span>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-warn" />
                  <span className="font-medium">Trial</span>
                </span>
                <span className="text-muted">
                  {remaining} / {trialQuestions} questions left
                </span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-line">
                <div
                  className="h-full rounded-full bg-brand transition-all"
                  style={{ width: `${(remaining / trialQuestions) * 100}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-muted">First {trialPages} pages of each document are indexed.</p>
            </>
          )}
        </div>
      </section>

      <section>
        <SectionLabel>Document</SectionLabel>
        {document ? (
          <div className="rounded-xl border border-line bg-surface p-3">
            <div className="flex items-start gap-3">
              <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-brand-soft text-brand">
                <FileText className="size-4.5" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold" title={document.name}>
                  {document.name}
                </div>
                <div className="mt-0.5 text-xs text-muted">
                  {document.pages < document.totalPages
                    ? `${document.pages} of ${document.totalPages} pages`
                    : `${document.pages} ${document.pages === 1 ? "page" : "pages"}`}{" "}
                  · {document.chunks} passages
                </div>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <SidebarButton onClick={onPickFile} disabled={busy}>
                <RefreshCw className="size-3.5" /> Replace
              </SidebarButton>
              <SidebarButton onClick={onRemoveDocument} disabled={busy}>
                <Trash2 className="size-3.5" /> Remove
              </SidebarButton>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={onPickFile}
            disabled={busy}
            className="flex w-full items-center gap-3 rounded-xl border border-dashed border-line bg-surface/60 p-3 text-left text-sm transition hover:border-brand hover:bg-surface disabled:opacity-50"
          >
            <Upload className="size-4 text-brand" />
            <span>
              <span className="font-medium">Upload a document</span>
              <span className="block text-xs text-muted">PDF or Word (.docx)</span>
            </span>
          </button>
        )}
      </section>

      <p className="mt-auto text-xs leading-relaxed text-muted">
        Answers come only from the passages retrieved from your document, with sources shown under each answer.
      </p>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <h2 className="mb-2.5 text-[0.7rem] font-bold tracking-[0.12em] text-muted uppercase">{children}</h2>;
}

function SidebarButton({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }) {
  return (
    <button
      type="button"
      {...props}
      className="flex items-center justify-center gap-1.5 rounded-lg border border-line py-1.5 text-xs font-medium transition hover:border-brand hover:text-brand disabled:pointer-events-none disabled:opacity-50"
    >
      {children}
    </button>
  );
}

export function Notice({ tone, children }: { tone: "danger" | "warn"; children: React.ReactNode }) {
  const styles = tone === "danger" ? "bg-danger-soft text-danger" : "bg-warn-soft text-warn";
  return (
    <div className={`flex gap-2.5 rounded-xl p-3 text-sm leading-relaxed ${styles}`} role="alert">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>{children}</div>
    </div>
  );
}
