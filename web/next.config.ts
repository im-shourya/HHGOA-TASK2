import type { NextConfig } from "next";

/**
 * Static export on purpose.
 *
 * The whole product ships as one service: `next build` emits `web/out`, and the
 * FastAPI process serves it alongside `/api/*`. One container, one URL, no CORS,
 * no second deploy to keep in sync — and the UI keeps working even if it is
 * later handed to a CDN, because every byte of it is static.
 *
 * In development `next dev` runs on :3000 and talks to the API on :8000 via
 * NEXT_PUBLIC_API_BASE (see .env.development).
 *
 * Linting is not configured here: Next 16 dropped the `eslint` key from
 * NextConfig and no longer runs ESLint during `next build`. `npm run lint` is a
 * separate step (and a separate CI gate) rather than a silent build-time one.
 */
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // Emit `out/index.html` (not `out/index/index.html`) so a plain static file
  // server resolves `/` without rewrite rules.
  trailingSlash: false,
};

export default nextConfig;
