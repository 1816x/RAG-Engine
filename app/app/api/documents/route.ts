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
    const { title, text } = await req.json();
    if (!title || !text) {
      return NextResponse.json({ error: "title and text are required" }, { status: 400 });
    }
    return NextResponse.json(await addDocument(title, text));
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown error" },
      { status: 502 },
    );
  }
}
