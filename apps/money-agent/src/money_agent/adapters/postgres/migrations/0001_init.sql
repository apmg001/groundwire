-- Initial schema for Groundwire Money.
--
-- Every table is keyed by user_id first: all queries filter on it, and it is
-- the natural shard key if the data ever outgrows one database.
-- Money is stored as whole paise (BIGINT), never floating point.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id          TEXT PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 128),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE transactions (
    id             UUID PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    posted_on      DATE NOT NULL,
    description    TEXT NOT NULL CHECK (length(description) BETWEEN 1 AND 300),
    amount_paise   BIGINT NOT NULL CHECK (amount_paise > 0),
    direction      TEXT NOT NULL CHECK (direction IN ('debit', 'credit')),
    category       TEXT NOT NULL,
    balance_paise  BIGINT,
    fingerprint    TEXT NOT NULL,
    imported_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-uploading a statement inserts nothing twice.
    CONSTRAINT transactions_user_fingerprint UNIQUE (user_id, fingerprint)
);

-- Newest-first listing and keyset pagination: (user_id, posted_on DESC, id DESC).
CREATE INDEX transactions_user_date ON transactions (user_id, posted_on DESC, id DESC);
-- Per-category queries over a period.
CREATE INDEX transactions_user_category ON transactions (user_id, category, posted_on);

CREATE TABLE memories (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('goal', 'fact', 'preference')),
    content     TEXT NOT NULL CHECK (length(content) BETWEEN 1 AND 500),
    -- Must match MONEY_EMBEDDING_DIM (checked at startup).
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL
);

-- Similarity search is exact and per user: a user has at most a few hundred
-- memories, so scanning them (found through this index) is fast and never
-- misses a result. An approximate HNSW index would only pay off for search
-- across all users, which the application never does.
CREATE INDEX memories_user_kind ON memories (user_id, kind, updated_at DESC);

CREATE TABLE conversations (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX conversations_user ON conversations (user_id, updated_at DESC);

CREATE TABLE messages (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    meta             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL
);

CREATE INDEX messages_conversation ON messages (conversation_id, id DESC);
