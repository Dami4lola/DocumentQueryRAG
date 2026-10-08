"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Menu, X } from "lucide-react";

import { ChatView } from "@/components/ChatView";
import { EmptyState } from "@/components/EmptyState";
import { IngestProgress, type IngestState } from "@/components/IngestProgress";
import type { ChatMessage } from "@/components/Message";
import { Notice, Sidebar } from "@/components/Sidebar";
import {
  ApiRequestError,
  askQuestion,
  deleteDocument,
  extractDocumentPages,
  fetchConfig,
  ingestDocument,
} from "@/lib/api";
import { ACCEPTED_FILES } from "@/lib/documents";
import type { ConfigResponse, DocumentInfo } from "@/lib/types";

const USER_KEY_STORAGE = "papertrail:openai-key";

export function Papertrail() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [userKey, setUserKey] = useState("");
  const [document, setDocument] = useState<DocumentInfo | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [questionsUsed, setQuestionsUsed] = useState(0);
  const [ingest, setIngest] = useState<IngestState | null>(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const tokenRef = useRef<string | null>(null);

  useEffect(() => {
    tokenRef.current = token;
  }, [token]);

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setError("Could not reach the server. Refresh the page to try again."));
    try {
      // Read after hydration: the server-rendered page has no access to sessionStorage.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setUserKey(sessionStorage.getItem(USER_KEY_STORAGE) ?? "");
    } catch {
      // Storage can be unavailable (private mode, blocked site data); the key just won't persist.
    }
  }, []);

  // Drop the document's vectors when the tab closes, since its session can't be resumed.
  useEffect(() => {
    const onPageHide = (event: PageTransitionEvent) => {
      if (!event.persisted && tokenRef.current) deleteDocument(tokenRef.current);
    };
    window.addEventListener("pagehide", onPageHide);
    return () => window.removeEventListener("pagehide", onPageHide);
  }, []);

  const updateUserKey = useCallback((key: string) => {
    setUserKey(key);
    try {
      if (key) sessionStorage.setItem(USER_KEY_STORAGE, key);
      else sessionStorage.removeItem(USER_KEY_STORAGE);
    } catch {}
  }, []);

  const unlimited = Boolean(userKey.trim());
  const trialQuestions = config?.trialQuestions ?? 2;
  const trialExhausted = !unlimited && questionsUsed >= trialQuestions;
  const missingPinecone = config ? !config.hasPinecone : false;
  const needsKey = config ? !config.hasOpenAIKey && !unlimited : false;

  async function processFile(file: Blob, name: string) {
    if (ingest) return;
    if (missingPinecone) {
      setError("The server isn't connected to Pinecone yet, so documents can't be indexed.");
      return;
    }
    if (needsKey) {
      setError("Add your OpenAI API key in the sidebar to process a document.");
      setSidebarOpen(true);
      return;
    }
    setError(null);
    setSidebarOpen(false);
    setIngest({ fileName: name, stage: "reading" });
    try {
      const { pages, totalPages } = await extractDocumentPages(file, name);
      for await (const event of ingestDocument({ name, pages, totalPages, previousToken: token }, userKey.trim())) {
        if (event.type === "progress") {
          setIngest(event.stage === "embedding" ? { fileName: name, ...event } : { fileName: name, stage: "indexing" });
        } else if (event.type === "ready") {
          setDocument(event.document);
          setToken(event.token);
          setMessages([]);
          setQuestionsUsed(event.questionsUsed);
        } else if (event.type === "error") {
          throw new ApiRequestError(event.message);
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not process this document.");
    } finally {
      setIngest(null);
    }
  }

  function onFileChosen(file: File | undefined) {
    if (!file) return;
    processFile(file, file.name);
  }

  async function loadSample() {
    try {
      const response = await fetch("/sample.pdf");
      if (!response.ok) throw new Error();
      await processFile(await response.blob(), "sample.pdf");
    } catch {
      setError("Could not load the included sample document.");
    }
  }

  function removeDocument() {
    if (token) deleteDocument(token);
    setDocument(null);
    setToken(null);
    setMessages([]);
    setQuestionsUsed(0);
  }

  function patchMessage(id: string, patch: Partial<ChatMessage> | ((message: ChatMessage) => Partial<ChatMessage>)) {
    setMessages((current) =>
      current.map((message) =>
        message.id === id ? { ...message, ...(typeof patch === "function" ? patch(message) : patch) } : message,
      ),
    );
  }

  async function ask(question: string) {
    if (!token || asking || trialExhausted) return;
    setError(null);
    const answerId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: question },
      { id: answerId, role: "assistant", content: "", status: "thinking" },
    ]);
    setAsking(true);
    try {
      for await (const event of askQuestion(question, token, userKey.trim())) {
        if (event.type === "meta") {
          setToken(event.token);
          setQuestionsUsed(event.questionsUsed);
          patchMessage(answerId, { sources: event.sources, status: "streaming" });
        } else if (event.type === "delta") {
          patchMessage(answerId, (message) => ({ content: message.content + event.text }));
        } else if (event.type === "done") {
          patchMessage(answerId, { status: "done" });
        } else if (event.type === "error") {
          patchMessage(answerId, { status: "error", content: event.message, sources: undefined });
        }
      }
    } catch (caught) {
      if (caught instanceof ApiRequestError && caught.code === "trial_exhausted") {
        setQuestionsUsed(trialQuestions);
      }
      if (caught instanceof ApiRequestError && caught.code === "invalid_session") {
        setDocument(null);
        setToken(null);
      }
      patchMessage(answerId, {
        status: "error",
        content: caught instanceof Error ? caught.message : "Could not answer this question.",
      });
    } finally {
      setAsking(false);
    }
  }

  function focusKeyInput() {
    setSidebarOpen(true);
    // Wait for the drawer to open on small screens before focusing.
    setTimeout(() => window.document.getElementById("openai-key")?.focus(), 50);
  }

  return (
    <div className="flex h-dvh overflow-hidden">
      <input
        ref={fileInput}
        type="file"
        accept={ACCEPTED_FILES}
        className="hidden"
        onChange={(event) => {
          onFileChosen(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      {sidebarOpen && (
        <div className="fixed inset-0 z-30 bg-black/30 lg:hidden" onClick={() => setSidebarOpen(false)} aria-hidden />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-[300px] max-w-[85vw] border-r border-line bg-sidebar transition-transform lg:static lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          config={config}
          userKey={userKey}
          onUserKeyChange={updateUserKey}
          document={document}
          busy={Boolean(ingest) || asking}
          questionsUsed={questionsUsed}
          onPickFile={() => fileInput.current?.click()}
          onRemoveDocument={removeDocument}
          onClose={() => setSidebarOpen(false)}
        />
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-line px-4 py-3 lg:hidden">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg p-1.5 hover:bg-line/60"
            aria-label="Open sidebar"
          >
            <Menu className="size-5" />
          </button>
          <span className="font-display text-lg font-extrabold tracking-tight text-brand">
            papertrail<span className="text-muted/60">.</span>
          </span>
          <span className="ml-auto rounded-full bg-brand-soft px-2.5 py-1 text-xs font-medium text-brand">
            {unlimited ? "Unlimited" : `Trial · ${Math.max(0, trialQuestions - questionsUsed)} left`}
          </span>
        </header>

        {error && (
          <div className="mx-auto w-full max-w-3xl px-4 pt-4 sm:px-8">
            <div className="relative">
              <Notice tone="danger">{error}</Notice>
              <button
                type="button"
                onClick={() => setError(null)}
                className="absolute top-2.5 right-2.5 rounded p-0.5 text-danger hover:bg-danger/10"
                aria-label="Dismiss"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>
        )}

        {ingest ? (
          <div className="flex-1 overflow-y-auto">
            <IngestProgress state={ingest} />
          </div>
        ) : document ? (
          <ChatView
            document={document}
            messages={messages}
            busy={asking}
            trialExhausted={trialExhausted}
            onAsk={ask}
            onAddKey={focusKeyInput}
          />
        ) : (
          <div className="flex-1 overflow-y-auto">
            <EmptyState
              disabled={Boolean(ingest)}
              onPickFile={() => fileInput.current?.click()}
              onFile={onFileChosen}
              onSample={loadSample}
            />
          </div>
        )}
      </main>
    </div>
  );
}
