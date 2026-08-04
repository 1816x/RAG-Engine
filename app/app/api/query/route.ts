import { NextResponse } from "next/server";
import { query } from "@/app/lib/rag";
import { INPUT_LIMITS, readLimitedJson } from "@/app/lib/limits";
import { apiErrorResponse } from "@/app/lib/api-errors";

export async function POST(req: Request) {
  try {
    const body = await readLimitedJson<{ question?: unknown; k?: unknown }>(req);
    if (typeof body.question !== "string" || !body.question.trim()) {
      return NextResponse.json({ error: "question is required" }, { status: 400 });
    }
    if (body.question.trim().length > INPUT_LIMITS.questionChars) {
      return NextResponse.json(
        { error: `question must be at most ${INPUT_LIMITS.questionChars} characters` },
        { status: 422 },
      );
    }

    const result = await query(
      body.question.trim(),
      typeof body.k === "number" ? body.k : 5,
    );
    return NextResponse.json(result);
  } catch (err) {
    return apiErrorResponse(err);
  }
}
