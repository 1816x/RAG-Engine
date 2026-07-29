import { NextResponse } from "next/server";
import { query } from "@/app/lib/rag";

export async function POST(req: Request) {
  try {
    const { question, k } = await req.json();
    if (!question || typeof question !== "string") {
      return NextResponse.json({ error: "question is required" }, { status: 400 });
    }
    const result = await query(question, k ?? 5);
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown error" },
      { status: 502 },
    );
  }
}
