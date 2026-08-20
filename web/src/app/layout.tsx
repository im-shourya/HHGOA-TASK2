import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice RAG · MSMARCO-XI",
  description:
    "Speak a question in English or Hindi: transcribed, retrieved from a hybrid index over MSMARCO-XI, answered with citations and a grounding check — core path under 200 ms.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="bg-ink-950 text-ink-100 min-h-full antialiased">
        {children}
      </body>
    </html>
  );
}
