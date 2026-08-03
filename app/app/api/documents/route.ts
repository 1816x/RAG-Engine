import { NextResponse } from "next/server";
import { addDocument, listDocuments } from "@/app/lib/rag";

export async function GET() {
  try {
    return NextResponse.json(await listDocuments());
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown error" },
      { status: 502 },
    );
  }
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { title?: unknown; text?: unknown };
    if (typeof body.title !== "string" || !body.title.trim()) {
      return NextResponse.json({ error: "title is required" }, { status: 400 });
    }
    if (typeof body.text !== "string" || !body.text.trim()) {
      return NextResponse.json({ error: "document text is required" }, { status: 400 });
    }

    return NextResponse.json(await addDocument(body.title.trim(), body.text));
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown error" },
      { status: 502 },
    );
  }
}
