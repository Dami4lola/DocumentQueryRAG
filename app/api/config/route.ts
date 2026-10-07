import { connection } from "next/server";

import { serverConfig, TRIAL_PAGES, TRIAL_QUESTIONS } from "@/lib/rag";
import type { ConfigResponse } from "@/lib/types";

export async function GET() {
  // Read env vars at request time rather than baking them in at build time.
  await connection();
  return Response.json({
    ...serverConfig(),
    trialQuestions: TRIAL_QUESTIONS,
    trialPages: TRIAL_PAGES,
  } satisfies ConfigResponse);
}
