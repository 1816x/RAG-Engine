import { NextResponse } from "next/server";
import { getStats } from "@/app/lib/rag";
import { apiErrorResponse } from "@/app/lib/api-errors";

export async function GET() {
  try {
    return NextResponse.json(await getStats());
  } catch (err) {
    return apiErrorResponse(err);
  }
}
