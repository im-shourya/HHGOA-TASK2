import type { Metadata } from "next";
import { Oswald, Space_Mono } from "next/font/google";
import "./globals.css";

const oswald = Oswald({
  subsets: ["latin"],
  variable: "--font-oswald",
});

const spaceMono = Space_Mono({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-space-mono",
});

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
      <body
        className={`${oswald.variable} ${spaceMono.variable} bg-goa-green text-goa-cream min-h-full antialiased font-sans`}
      >
        {children}
      </body>
    </html>
  );
}
