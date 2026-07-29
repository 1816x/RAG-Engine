import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Engine",
  description: "Ask questions over a document set, backed by a from-scratch HNSW index.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
