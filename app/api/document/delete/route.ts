import { deleteNamespace } from "@/lib/rag";
import { verifySession } from "@/lib/session";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const session = verifySession(body?.token);
  if (session) await deleteNamespace(session.ns);
  return new Response(null, { status: 204 });
}
