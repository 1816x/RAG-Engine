import { NextResponse } from "next/server";
import { addDocument, listDocuments } from "@/app/lib/rag";
import { INPUT_LIMITS, readLimitedJson } from "@/app/lib/limits";
import { apiErrorResponse } from "@/app/lib/api-errors";

export async function GET() {
  try {
    return NextResponse.json(await listDocuments());
  } catch (err) {
    return apiErrorResponse(err);
  }
}

export async function POST(req: Request) {
  try {
    const body = await readLimitedJson<{ title?: unknown; text?: unknown }>(req);
    if (typeof body.title !== "string" || !body.title.trim()) {
      return NextResponse.json({ error: "title is required" }, { status: 400 });
    }
    if (body.title.trim().length > INPUT_LIMITS.titleChars) {
      return NextResponse.json(
        { error: `title must be at most ${INPUT_LIMITS.titleChars} characters` },
        { status: 422 },
      );
    }
    if (typeof body.text !== "string" || !body.text.trim()) {
      return NextResponse.json({ error: "document text is required" }, { status: 400 });
    }
    if (body.text.length > INPUT_LIMITS.documentChars) {
      return NextResponse.json(
        { error: `document text must be at most ${INPUT_LIMITS.documentChars} characters` },
        { status: 422 },
      );
    }

    return NextResponse.json(await addDocument(body.title.trim(), body.text));
  } catch (err) {
    return apiErrorResponse(err);
  }
}
