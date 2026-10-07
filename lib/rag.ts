import "server-only";

import OpenAI from "openai";
import { Pinecone, type Index } from "@pinecone-database/pinecone";

export const EMBEDDING_MODEL = "text-embedding-3-small";
export const CHAT_MODEL = "gpt-4o-mini";
export const CHUNK_SIZE = 1000;
export const CHUNK_STRIDE = 800;
export const TOP_K = 3;

export const TRIAL_QUESTIONS = 2;
export const TRIAL_PAGES = 5;
// Guards against oversized requests; the browser sends extracted text, not the PDF.
export const MAX_CHARS_PER_PAGE_TRIAL = 10_000;
export const MAX_TOTAL_CHARS = 1_000_000;

const EMBED_BATCH = 100;
const UPSERT_BATCH = 100;

export const SYSTEM_PROMPT =
  "Answer the user's question using only the supplied document passages. " +
  "If the passages do not contain the answer, say so plainly.";

type PassageMetadata = { text: string };

export function serverConfig() {
  const missing: string[] = [];
  if (!process.env.OPENAI_API_KEY) missing.push("OPENAI_API_KEY");
  if (!process.env.PINECONE_API_KEY) missing.push("PINECONE_API_KEY");
  if (!process.env.PINECONE_INDEX_NAME) missing.push("PINECONE_INDEX_NAME");
  return {
    hasOpenAIKey: Boolean(process.env.OPENAI_API_KEY),
    hasPinecone: Boolean(process.env.PINECONE_API_KEY && process.env.PINECONE_INDEX_NAME),
    missing,
  };
}

export function getOpenAI(userKey: string | null): OpenAI {
  const apiKey = userKey || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new UserFacingError("Add an OpenAI key in the sidebar, or configure OPENAI_API_KEY on the server.");
  }
  return new OpenAI({ apiKey });
}

export function getIndex(namespace: string): Index<PassageMetadata> {
  const apiKey = process.env.PINECONE_API_KEY;
  const name = process.env.PINECONE_INDEX_NAME;
  if (!apiKey || !name) {
    throw new UserFacingError("The server is missing PINECONE_API_KEY or PINECONE_INDEX_NAME.");
  }
  return new Pinecone({ apiKey }).index<PassageMetadata>({ name, namespace });
}

export function chunkText(text: string): string[] {
  const chunks: string[] = [];
  for (let start = 0; start < text.length; start += CHUNK_STRIDE) {
    const chunk = text.slice(start, start + CHUNK_SIZE);
    if (chunk.trim()) chunks.push(chunk);
  }
  return chunks;
}

export async function embed(client: OpenAI, inputs: string[]): Promise<number[][]> {
  const response = await client.embeddings.create({ model: EMBEDDING_MODEL, input: inputs });
  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}

/** Embeds all chunks in batches, reporting progress after each batch. */
export async function embedChunks(
  client: OpenAI,
  chunks: string[],
  onProgress: (done: number) => void,
): Promise<number[][]> {
  const vectors: number[][] = [];
  for (let start = 0; start < chunks.length; start += EMBED_BATCH) {
    vectors.push(...(await embed(client, chunks.slice(start, start + EMBED_BATCH))));
    onProgress(vectors.length);
  }
  return vectors;
}

export async function upsertChunks(namespace: string, chunks: string[], vectors: number[][]) {
  const index = getIndex(namespace);
  for (let start = 0; start < chunks.length; start += UPSERT_BATCH) {
    await index.upsert({
      records: chunks.slice(start, start + UPSERT_BATCH).map((text, offset) => ({
        id: `${namespace}_${start + offset}`,
        values: vectors[start + offset],
        metadata: { text },
      })),
    });
  }
}

export async function retrieve(client: OpenAI, namespace: string, question: string): Promise<string[]> {
  const [vector] = await embed(client, [question]);
  const results = await getIndex(namespace).query({ vector, topK: TOP_K, includeMetadata: true });
  return results.matches
    .map((match) => match.metadata?.text)
    .filter((text): text is string => Boolean(text));
}

export async function deleteNamespace(namespace: string) {
  try {
    await getIndex(namespace).deleteAll();
  } catch (error) {
    // A namespace that was never created (or already removed) is not an error for us.
    console.warn(`Could not delete namespace ${namespace}:`, describeError(error));
  }
}

/** An error whose message is safe and useful to show in the UI. */
export class UserFacingError extends Error {
  constructor(message: string, readonly status = 400) {
    super(message);
  }
}

export function describeError(error: unknown): string {
  if (error instanceof UserFacingError) return error.message;
  if (error instanceof OpenAI.APIError) {
    if (error.status === 401) return "OpenAI rejected the API key. Check the key and try again.";
    if (error.status === 429) return "OpenAI rate limit or quota reached. Try again shortly, or use a different key.";
    return `OpenAI error: ${error.message}`;
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}
