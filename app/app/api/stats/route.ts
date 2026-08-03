import { NextResponse } from "next/server";
import { getStats } from "@/app/lib/rag";

export async function GET() {
  try {
    return NextResponse.json(await getStats());
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "unknown error" },
      { status: 502 },
    );
  }
}
