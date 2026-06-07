# Migration runner image for the kamaji diary database.
# Built & pushed by .github/workflows/migrate-image.yml as
# ghcr.io/ed-yahska-xyz/blogs-migrate, and run as a one-shot `migrate`
# service in the kamaji docker-compose stack (gates the kamaji app via
# depends_on: service_completed_successfully).
#
# Requires DATABASE_URL at runtime; applies db/migrations/*.sql idempotently
# (tracked in the _migrations table) and exits 0.
FROM oven/bun:1
WORKDIR /db

# Install deps first for layer caching (only `postgres` in production).
COPY package.json ./
RUN bun install --production

# Migration runner + SQL files.
COPY . .

CMD ["bun", "run", "migrate"]
