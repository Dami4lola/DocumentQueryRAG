import { ndjsonResponse } from "@/lib/ndjson";
import { CHAT_MODEL, describeError, getOpenAI, retrieve, SYSTEM_PROMPT, TRIAL_QUESTIONS } from "@/lib/rag";
import { signSession, verifySession } from "@/lib/session";
import { USER_KEY_HEADER, type ApiError, type AskEvent, type AskRequest } from "@/lib/types";

export const maxDuration = 60;

const NO_SOURCES_ANSWER = "I couldn't find a relevant passage in this document to answer that question.";

export async function POST(request: Request) {
  let body: AskRequest;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid request body." } satisfies ApiError, { status: 400 });
  }

  const question = typeof body.question === "string" ? body.question.trim().slice(0, 2000) : "";
  if (!question) {
    return Response.json({ error: "Ask a question first." } satisfies ApiError, { status: 400 });
  }

  const session = verifySession(body.token);
  if (!session) {
    return Response.json(
      { error: "This document session is no longer valid. Upload the document again.", code: "invalid_session" } satisfies ApiError,
      { status: 401 },
    );
  }

  const userKey = request.headers.get(USER_KEY_HEADER)?.trim() || null;
  if (!userKey && session.q >= TRIAL_QUESTIONS) {
    return Response.json(
      {
        error: "Your two-question trial is complete. Add your own OpenAI API key to keep asking.",
        code: "trial_exhausted",
      } satisfies ApiError,
      { status: 403 },
    );
  }

  let client;
  try {
    client = getOpenAI(userKey);
  } catch (error) {
    return Response.json({ error: describeError(error) } satisfies ApiError, { status: 503 });
  }

  const updated = userKey ? session : { ...session, q: session.q + 1 };

  return ndjsonResponse<AskEvent>(async (send) => {
    try {
      const sources = await retrieve(client, session.ns, question);
      send({ type: "meta", sources, token: signSession(updated), questionsUsed: updated.q });

      if (sources.length === 0) {
        send({ type: "delta", text: NO_SOURCES_ANSWER });
        send({ type: "done" });
        return;
      }

      const stream = await client.chat.completions.create({
        model: CHAT_MODEL,
        temperature: 0,
        stream: true,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: `DOCUMENT PASSAGES:\n${sources.join("\n---\n")}\n\nQUESTION:\n${question}` },
        ],
      });
      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content;
        if (text) send({ type: "delta", text });
      }
      send({ type: "done" });
    } catch (error) {
      send({ type: "error", message: `Could not answer this question: ${describeError(error)}` });
    }
  });
}
