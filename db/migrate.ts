#!/usr/bin/env bun
import { readdir, readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import postgres from "postgres";

const here = dirname(fileURLToPath(import.meta.url));
const migrationsDir = join(here, "migrations");

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL is required");
  process.exit(1);
}

const sql = postgres(url, { onnotice: () => {} });

await sql`
  CREATE TABLE IF NOT EXISTS _migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
  )
`;

const applied = new Set(
  (await sql<{ name: string }[]>`SELECT name FROM _migrations`).map((r) => r.name),
);

const files = (await readdir(migrationsDir))
  .filter((f) => f.endsWith(".sql"))
  .sort();

let ran = 0;
for (const file of files) {
  if (applied.has(file)) continue;
  const body = await readFile(join(migrationsDir, file), "utf8");
  console.log(`applying ${file}`);
  await sql.begin(async (tx) => {
    await tx.unsafe(body);
    await tx`INSERT INTO _migrations (name) VALUES (${file})`;
  });
  ran++;
}

console.log(ran === 0 ? "no pending migrations" : `applied ${ran} migration(s)`);
await sql.end();
