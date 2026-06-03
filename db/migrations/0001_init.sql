CREATE TABLE diary_entries (
  id          BIGSERIAL PRIMARY KEY,
  entry_date  DATE NOT NULL UNIQUE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE paragraphs (
  id          BIGSERIAL PRIMARY KEY,
  entry_id    BIGINT NOT NULL REFERENCES diary_entries(id) ON DELETE CASCADE,
  body        TEXT NOT NULL,
  position    INT  NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX paragraphs_entry_id_idx ON paragraphs(entry_id, position);

CREATE TABLE paragraph_hashtags (
  paragraph_id BIGINT NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
  tag          TEXT   NOT NULL,
  PRIMARY KEY (paragraph_id, tag)
);
CREATE INDEX paragraph_hashtags_tag_idx ON paragraph_hashtags(tag);
