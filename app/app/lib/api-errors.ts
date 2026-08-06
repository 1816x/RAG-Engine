import { NextResponse } from "next/server";
import { InputRequestError } from "./limits";
import { RagServiceError } from "./rag";

export function apiErrorResponse(error: unknown) {
  const status =
    error instanceof InputRequestError || error instanceof RagServiceError
      ? error.status
      : 502;
  return NextResponse.json(
    { error: error instanceof Error ? error.message : "unknown error" },
    { status },
  );
}
