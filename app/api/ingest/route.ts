import { randomUUID } from "node:crypto";

import { ndjsonResponse } from "@/lib/ndjson";
import {
  chunkText,
  deleteNamespace,
  describeError,
  embedChunks,
  getIndex,
  getOpenAI,
  MAX_CHARS_PER_PAGE_TRIAL,
  MAX_TOTAL_CHARS,
  TRIAL_PAGES,
  upsertChunks,
} from "@/lib/rag";
import { signSession, verifySession } from "@/lib/session";
import { USER_KEY_HEADER, type ApiError, type IngestEvent, type IngestRequest } from "@/lib/types";

export const maxDuration = 60;

export async function POST(request: Request) {
  let body: IngestRequest;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid request body." } satisfies ApiError, { status: 400 });
  }

  const userKey = request.headers.get(USER_KEY_HEADER)?.trim() || null;
  const trial = !userKey;
  const name = typeof body.name === "string" && body.name ? body.name.slice(0, 200) : "document.pdf";
  if (!Array.isArray(body.pages) || body.pages.some((page) => typeof page !== "string")) {
    return Response.json({ error: "Invalid request body." } satisfies ApiError, { status: 400 });
  }

  let pages = trial ? body.pages.slice(0, TRIAL_PAGES) : body.pages;
  if (trial) pages = pages.map((page) => page.slice(0, MAX_CHARS_PER_PAGE_TRIAL));
  const rawText = pages.join("\n");
  if (!rawText.trim()) {
    return Response.json(
      { error: "No selectable text was found in this PDF. Scanned, image-only PDFs aren't supported." } satisfies ApiError,
      { status: 400 },
    );
  }
  if (rawText.length > MAX_TOTAL_CHARS) {
    return Response.json(
      { error: "This document is too large for the demo. Try a shorter PDF." } satisfies ApiError,
      { status: 413 },
    );
  }

  let client;
  try {
    client = getOpenAI(userKey);
    getIndex("probe"); // fail fast if Pinecone isn't configured
  } catch (error) {
    return Response.json({ error: describeError(error) } satisfies ApiError, { status: 503 });
  }

  const totalPages = Math.max(Number(body.totalPages) || pages.length, pages.length);
  const previous = verifySession(body.previousToken);
  const chunks = chunkText(rawText);

  return ndjsonResponse<IngestEvent>(async (send) => {
    const namespace = randomUUID();
    try {
      send({ type: "progress", stage: "embedding", done: 0, total: chunks.length });
      const vectors = await embedChunks(client, chunks, (done) =>
        send({ type: "progress", stage: "embedding", done, total: chunks.length }),
      );

      send({ type: "progress", stage: "indexing" });
      await upsertChunks(namespace, chunks, vectors);
      if (previous) await deleteNamespace(previous.ns);

      const document = { name, pages: pages.length, totalPages, chunks: chunks.length };
      // Trial questions carry over when a document is replaced, as they did per session before.
      const questionsUsed = previous?.q ?? 0;
      send({ type: "ready", token: signSession({ ns: namespace, ...document, q: questionsUsed }), document, questionsUsed });
    } catch (error) {
      await deleteNamespace(namespace);
      send({ type: "error", message: `Could not process this PDF: ${describeError(error)}` });
    }
  });
}
