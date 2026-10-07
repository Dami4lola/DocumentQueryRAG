import "server-only";

import { createHash, createHmac, timingSafeEqual } from "node:crypto";

/**
 * Stateless session for one indexed document. The token is signed so a client can only
 * query or delete namespaces this server created, and can't reset its own question count
 * without starting a new document.
 */
export type Session = {
  ns: string; // Pinecone namespace
  name: string;
  pages: number; // pages indexed
  totalPages: number;
  chunks: number;
  q: number; // trial questions used
};

function secret(): string {
  if (process.env.SESSION_SECRET) return process.env.SESSION_SECRET;
  // Derive a stable secret from another server-only value so local setup needs one less variable.
  const fallback = process.env.PINECONE_API_KEY;
  if (!fallback) throw new Error("Set SESSION_SECRET (or PINECONE_API_KEY) on the server.");
  return createHash("sha256").update(`papertrail-session:${fallback}`).digest("hex");
}

function sign(payload: string): string {
  return createHmac("sha256", secret()).update(payload).digest("base64url");
}

export function signSession(session: Session): string {
  const payload = Buffer.from(JSON.stringify(session)).toString("base64url");
  return `${payload}.${sign(payload)}`;
}

export function verifySession(token: unknown): Session | null {
  if (typeof token !== "string") return null;
  const [payload, signature] = token.split(".");
  if (!payload || !signature) return null;

  const expected = Buffer.from(sign(payload));
  const actual = Buffer.from(signature);
  if (expected.length !== actual.length || !timingSafeEqual(expected, actual)) return null;

  try {
    return JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as Session;
  } catch {
    return null;
  }
}
