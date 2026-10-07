"use client";

import { useState } from "react";
import { FileUp, MessageSquareText, Quote, Sparkles } from "lucide-react";

type Props = {
  disabled: boolean;
  onPickFile: () => void;
  onFile: (file: File) => void;
  onSample: () => void;
};

const STEPS = [
  { icon: FileUp, title: "Add a document", body: "Upload a PDF or use the ready-made sample." },
  { icon: MessageSquareText, title: "Ask naturally", body: "No keywords or special prompts required." },
  { icon: Quote, title: "Get grounded answers", body: "See the passages used to answer your question." },
];

export function EmptyState({ disabled, onPickFile, onFile, onSample }: Props) {
  const [dragging, setDragging] = useState(false);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-10 sm:px-8 sm:py-16">
      <p className="mb-3 text-xs font-bold tracking-[0.14em] text-brand uppercase">Your documents, understood</p>
      <h1 className="font-display text-4xl leading-[1.08] font-extrabold tracking-[-0.05em] text-balance sm:text-6xl">
        Good questions deserve clear answers.
      </h1>
      <p className="mt-4 max-w-xl text-lg text-muted">
        Chat with your PDF and get answers grounded in the document, with the passages behind every answer.
      </p>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files[0];
          if (file && !disabled) onFile(file);
        }}
        className={`mt-10 rounded-3xl border-2 border-dashed p-8 text-center transition sm:p-10 ${
          dragging ? "border-brand bg-brand-soft" : "border-line bg-surface"
        }`}
      >
        <div className="mx-auto grid size-14 place-items-center rounded-2xl bg-brand-soft text-brand">
          <FileUp className="size-6" />
        </div>
        <h2 className="mt-4 font-display text-xl font-bold tracking-tight">Drop a PDF here</h2>
        <p className="mt-1 text-sm text-muted">or choose one from your computer. Text-based PDFs only.</p>
        <div className="mt-6 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <button
            type="button"
            onClick={onPickFile}
            disabled={disabled}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-brand-hover disabled:opacity-50 sm:w-auto dark:text-[#0c1310]"
          >
            <FileUp className="size-4" /> Choose a PDF
          </button>
          <button
            type="button"
            onClick={onSample}
            disabled={disabled}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-surface px-5 py-2.5 text-sm font-semibold transition hover:border-brand hover:text-brand disabled:opacity-50 sm:w-auto"
          >
            <Sparkles className="size-4" /> Try the sample document
          </button>
        </div>
      </div>

      <h2 className="mt-12 mb-4 text-xs font-bold tracking-[0.12em] text-muted uppercase">How it works</h2>
      <ol className="grid gap-3 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, body }, index) => (
          <li key={title} className="rounded-2xl border border-line bg-surface p-4">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Icon className="size-4 text-brand" />
              <span className="text-muted">0{index + 1}</span> {title}
            </div>
            <p className="mt-1.5 text-sm text-muted">{body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
