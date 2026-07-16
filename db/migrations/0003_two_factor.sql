-- Better Auth two-factor (TOTP) schema, added when the twoFactor() plugin was
-- enabled in kamaji's src/services/auth/index.ts. Regenerate with:
--   bunx @better-auth/cli@latest generate --config src/services/auth/index.ts
-- (Identifiers stay quoted/camelCase to match Better Auth's Kysely queries.)

alter table "user" add column "twoFactorEnabled" boolean;

create table "twoFactor" (
  "id"          text not null primary key,
  "secret"      text not null,
  "backupCodes" text not null,
  "userId"      text not null references "user" ("id") on delete cascade,
  "verified"    boolean
);

create index "twoFactor_secret_idx" on "twoFactor" ("secret");
create index "twoFactor_userId_idx" on "twoFactor" ("userId");
