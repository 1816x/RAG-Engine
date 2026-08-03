import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Engine · Retrieval Workbench",
  description:
    "Index documents, inspect a from-scratch HNSW graph, and ask grounded questions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
