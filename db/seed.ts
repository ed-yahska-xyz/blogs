#!/usr/bin/env bun
import postgres from "postgres";

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL is required");
  process.exit(1);
}

const sql = postgres(url);

const today = new Date();
const iso = (offsetDays: number) => {
  const d = new Date(today);
  d.setDate(d.getDate() - offsetDays);
  return d.toISOString().split("T")[0]!;
};

const samples: { date: string; body: string }[] = [
  { date: iso(0), body: "First diary entry. Testing the system. #meta #setup" },
  { date: iso(0), body: "Second paragraph for today. Hashtags should be searchable. #search" },
  { date: iso(1), body: "Yesterday I shipped the schema. #postgres #ship" },
  { date: iso(3), body: "Three days ago: planning notes for the diary feature. #planning" },
];

const TAG_RE = /#([\w-]+)/g;
const extractTags = (body: string): string[] => {
  const tags = new Set<string>();
  for (const m of body.matchAll(TAG_RE)) tags.add(m[1]!.toLowerCase());
  return [...tags];
};

for (const { date, body } of samples) {
  await sql.begin(async (tx) => {
    const [entry] = await tx<{ id: number }[]>`
      INSERT INTO diary_entries (entry_date) VALUES (${date})
      ON CONFLICT (entry_date) DO UPDATE SET updated_at = now()
      RETURNING id
    `;
    const [{ next_pos }] = await tx<{ next_pos: number }[]>`
      SELECT COALESCE(MAX(position), -1) + 1 AS next_pos
      FROM paragraphs WHERE entry_id = ${entry!.id}
    `;
    const [p] = await tx<{ id: number }[]>`
      INSERT INTO paragraphs (entry_id, body, position)
      VALUES (${entry!.id}, ${body}, ${next_pos})
      RETURNING id
    `;
    const tags = extractTags(body);
    if (tags.length) {
      await tx`
        INSERT INTO paragraph_hashtags ${tx(tags.map((t) => ({ paragraph_id: p!.id, tag: t })))}
        ON CONFLICT DO NOTHING
      `;
    }
  });
}

console.log(`seeded ${samples.length} paragraph(s)`);
await sql.end();
