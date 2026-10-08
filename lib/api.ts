/** Browser-side helpers: document text extraction and calls to the API routes. */

import { documentKind, splitIntoPages } from "@/lib/documents";
import { readNdjson } from "@/lib/ndjson";
import {
  USER_KEY_HEADER,
  type ApiError,
  type AskEvent,
  type ConfigResponse,
  type IngestEvent,
  type IngestRequest,
} from "@/lib/types";

export class ApiRequestError extends Error {
  constructor(message: string, readonly code?: ApiError["code"]) {
    super(message);
  }
}

type ExtractedPages = { pages: string[]; totalPages: number };

export async function extractDocumentPages(file: Blob, name: string): Promise<ExtractedPages> {
  if (name.toLowerCase().endsWith(".doc")) {
    throw new ApiRequestError("Older .doc files aren't supported. Save it as .docx or PDF and try again.");
  }
  const kind = documentKind(name, file.type);
  if (kind === "pdf") return extractPdfPages(file);
  if (kind === "docx") return extractDocxPages(file);
  throw new ApiRequestError("Only PDF and Word (.docx) files are supported.");
}

async function extractDocxPages(file: Blob): Promise<ExtractedPages> {
  const mammoth = await import("mammoth");
  let text: string;
  try {
    ({ value: text } = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() }));
  } catch {
    throw new ApiRequestError("This file couldn't be read as a Word document.");
  }
  const pages = splitIntoPages(text);
  if (pages.length === 0) throw new ApiRequestError("No text was found in this document.");
  return { pages, totalPages: pages.length };
}

async function extractPdfPages(file: Blob): Promise<ExtractedPages> {
  const { getDocumentProxy, extractText } = await import("unpdf");
  let pdf;
  try {
    pdf = await getDocumentProxy(new Uint8Array(await file.arrayBuffer()));
  } catch {
    throw new ApiRequestError("This file couldn't be read as a PDF.");
  }
  const { totalPages, text } = await extractText(pdf, { mergePages: false });
  if (!text.join("").trim()) {
    throw new ApiRequestError("No selectable text was found in this PDF. Scanned, image-only PDFs aren't supported.");
  }
  return { pages: text, totalPages };
}

function headers(userKey: string): HeadersInit {
  return {
    "Content-Type": "application/json",
    ...(userKey ? { [USER_KEY_HEADER]: userKey } : {}),
  };
}

async function ensureOk(response: Response) {
  if (response.ok) return;
  const body = (await response.json().catch(() => null)) as ApiError | null;
  throw new ApiRequestError(body?.error ?? `Request failed (${response.status}).`, body?.code);
}

export async function fetchConfig(): Promise<ConfigResponse> {
  const response = await fetch("/api/config", { cache: "no-store" });
  await ensureOk(response);
  return response.json();
}

export async function* ingestDocument(body: IngestRequest, userKey: string): AsyncGenerator<IngestEvent> {
  const response = await fetch("/api/ingest", {
    method: "POST",
    headers: headers(userKey),
    body: JSON.stringify(body),
  });
  await ensureOk(response);
  yield* readNdjson<IngestEvent>(response);
}

export async function* askQuestion(
  question: string,
  token: string,
  userKey: string,
  signal?: AbortSignal,
): AsyncGenerator<AskEvent> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: headers(userKey),
    body: JSON.stringify({ question, token }),
    signal,
  });
  await ensureOk(response);
  yield* readNdjson<AskEvent>(response);
}

export function deleteDocument(token: string) {
  // keepalive lets this finish even if the page is being closed.
  return fetch("/api/document/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
    keepalive: true,
  }).catch(() => undefined);
}
