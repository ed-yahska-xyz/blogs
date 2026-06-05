# blogs/db

Schema, migrations, and seed scripts for the diary database used by [kamaji](https://github.com/ed-yahska-xyz/kamaji).

The database itself runs as a container in `../virtuals/kamaji/docker-compose.yml`. This directory owns the schema source of truth.

## Usage

```bash
cd db
bun install
export DATABASE_URL=postgres://kamaji:PASSWORD@localhost:5432/diary
bun run migrate   # applies any new SQL files in migrations/
bun run seed      # optional sample rows
```

## Adding a migration

Create `migrations/NNNN_short_name.sql` (zero-padded, monotonic). Files run in lexical order; each runs in its own transaction and is recorded in `_migrations`.

## Schema overview

- `diary_entries(id, entry_date UNIQUE, …)` — one row per calendar day
- `paragraphs(id, entry_id, body, position, …)` — ordered paragraphs within a day
- `paragraph_hashtags(paragraph_id, tag)` — denormalized `#tag` index extracted at write time
