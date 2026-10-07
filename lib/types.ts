/** Shapes shared between the API routes and the browser. */

export type DocumentInfo = {
  name: string;
  pages: number;
  totalPages: number;
  chunks: number;
};

export type ConfigResponse = {
  hasOpenAIKey: boolean;
  hasPinecone: boolean;
  missing: string[];
  trialQuestions: number;
  trialPages: number;
};

export type IngestRequest = {
  name: string;
  pages: string[];
  totalPages: number;
  previousToken?: string | null;
};

export type IngestEvent =
  | { type: "progress"; stage: "embedding"; done: number; total: number }
  | { type: "progress"; stage: "indexing" }
  | { type: "ready"; token: string; document: DocumentInfo; questionsUsed: number }
  | { type: "error"; message: string };

export type AskRequest = { question: string; token: string };

export type AskEvent =
  | { type: "meta"; sources: string[]; token: string; questionsUsed: number }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

export type ApiError = { error: string; code?: "trial_exhausted" | "invalid_session" };

export const USER_KEY_HEADER = "x-openai-key";
