"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { AlertCircle, ChevronDown, Quote } from "lucide-react";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  status?: "thinking" | "streaming" | "done" | "error";
};

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand px-4 py-2.5 whitespace-pre-wrap text-white dark:text-[#0c1310]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.status === "error") {
    return (
      <div className="flex max-w-[92%] gap-2.5 rounded-2xl rounded-bl-md bg-danger-soft px-4 py-3 text-sm text-danger" role="alert">
        <AlertCircle className="mt-0.5 size-4 shrink-0" />
        <div>{message.content}</div>
      </div>
    );
  }

  return (
    <div className="max-w-[92%]">
      <div className="rounded-2xl rounded-bl-md border border-line bg-surface px-4 py-3 leading-relaxed">
        {message.status === "thinking" ? (
          <span className="inline-flex items-center gap-1 py-1" aria-label="Searching the document">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="size-1.5 animate-bounce rounded-full bg-muted"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
          </span>
        ) : (
          <div className={`answer ${message.status === "streaming" ? "streaming-cursor" : ""}`}>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
      {message.sources && message.sources.length > 0 && <Sources sources={message.sources} />}
    </div>
  );
}

function Sources({ sources }: { sources: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1 text-xs font-medium text-muted transition hover:border-brand hover:text-brand"
      >
        <Quote className="size-3" />
        Sources · {sources.length} {sources.length === 1 ? "passage" : "passages"}
        <ChevronDown className={`size-3.5 transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <ol className="mt-2 space-y-2">
          {sources.map((source, index) => (
            <li key={index} className="rounded-xl border border-line bg-sidebar p-3">
              <div className="mb-1 text-[0.7rem] font-bold tracking-[0.1em] text-brand uppercase">
                Passage {index + 1}
              </div>
              <p className="text-sm leading-relaxed whitespace-pre-wrap text-muted">{source.trim()}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
