import { NextResponse } from "next/server";
import { deleteDocument, replaceDocument } from "@/app/lib/rag";
import { INPUT_LIMITS, readLimitedJson } from "@/app/lib/limits";
import { apiErrorResponse } from "@/app/lib/api-errors";

type Context = { params: Promise<{ id: string }> };

function documentId(raw: string): number | null {
  const id = Number(raw);
  return Number.isSafeInteger(id) && id >= 0 ? id : null;
}

export async function DELETE(_: Request, context: Context) {
  const id = documentId((await context.params).id);
  if (id === null) return NextResponse.json({ error: "invalid document id" }, { status: 422 });
  try {
    return NextResponse.json(await deleteDocument(id));
  } catch (err) {
    return apiErrorResponse(err);
  }
}

export async function PUT(req: Request, context: Context) {
  const id = documentId((await context.params).id);
  if (id === null) return NextResponse.json({ error: "invalid document id" }, { status: 422 });
  try {
    const body = await readLimitedJson<{ title?: unknown; text?: unknown }>(req);
    if (typeof body.title !== "string" || !body.title.trim()) {
      return NextResponse.json({ error: "title is required" }, { status: 422 });
    }
    if (body.title.trim().length > INPUT_LIMITS.titleChars) {
      return NextResponse.json({ error: `title must be at most ${INPUT_LIMITS.titleChars} characters` }, { status: 422 });
    }
    if (typeof body.text !== "string" || !body.text.trim()) {
      return NextResponse.json({ error: "document text is required" }, { status: 422 });
    }
    if (body.text.length > INPUT_LIMITS.documentChars) {
      return NextResponse.json({ error: `document text must be at most ${INPUT_LIMITS.documentChars} characters` }, { status: 422 });
    }
    return NextResponse.json(await replaceDocument(id, body.title.trim(), body.text));
  } catch (err) {
    return apiErrorResponse(err);
  }
}
