-- ============================================
-- SUPABASE SETUP FOR RAG WITH HYBRID SEARCH
-- ============================================
-- Embedding model: voyage-3-large (1024 dimensions)
-- Provider: Voyage AI
-- ============================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- TABLE FOR voyage-3-large (1024 dims)
-- ============================================

DROP TABLE IF EXISTS general_embeddings_1024 CASCADE;

CREATE TABLE general_embeddings_1024 (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    content_tsvector TSVECTOR,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX general_embeddings_1024_embedding_idx
ON general_embeddings_1024
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX general_embeddings_1024_tsvector_idx
ON general_embeddings_1024
USING GIN (content_tsvector);

-- ============================================
-- TRIGGER FOR AUTOMATIC TSVECTOR (SPANISH)
-- ============================================

CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('spanish', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON general_embeddings_1024;
CREATE TRIGGER tsvector_update
    BEFORE INSERT OR UPDATE ON general_embeddings_1024
    FOR EACH ROW
    EXECUTE FUNCTION update_content_tsvector();

-- ============================================
-- SEARCH FUNCTIONS
-- ============================================

-- Semantic search (embeddings only)
CREATE OR REPLACE FUNCTION rag_search_semantic(
    query_embedding VECTOR(1024),
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
    FROM general_embeddings_1024 d
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Full-text search (keywords only)
CREATE OR REPLACE FUNCTION rag_search_fulltext(
    query_text TEXT,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        ts_rank(d.content_tsvector, plainto_tsquery('spanish', query_text))::FLOAT AS rank
    FROM general_embeddings_1024 d
    WHERE d.content_tsvector @@ plainto_tsquery('spanish', query_text)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Hybrid search with RRF (Reciprocal Rank Fusion)
CREATE OR REPLACE FUNCTION rag_hybrid_search_user_context(
    query_text TEXT,
    query_embedding VECTOR(1024),
    match_count INT DEFAULT 5,
    semantic_weight FLOAT DEFAULT 0.5,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    semantic_rank INT,
    fulltext_rank INT,
    rrf_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT
            d.id,
            d.content,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding) AS rank
        FROM general_embeddings_1024 d
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    fulltext_results AS (
        SELECT
            d.id,
            d.content,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(d.content_tsvector, plainto_tsquery('spanish', query_text)) DESC
            ) AS rank
        FROM general_embeddings_1024 d
        WHERE d.content_tsvector @@ plainto_tsquery('spanish', query_text)
        LIMIT match_count * 2
    ),
    combined AS (
        SELECT
            COALESCE(s.id, f.id) AS id,
            COALESCE(s.content, f.content) AS content,
            COALESCE(s.rank, 1000)::INT AS semantic_rank,
            COALESCE(f.rank, 1000)::INT AS fulltext_rank,
            ((semantic_weight * (1.0 / (rrf_k + COALESCE(s.rank, 1000)))) +
            ((1 - semantic_weight) * (1.0 / (rrf_k + COALESCE(f.rank, 1000)))))::FLOAT AS rrf_score
        FROM semantic_results s
        FULL OUTER JOIN fulltext_results f ON s.id = f.id
    )
    SELECT
        c.id,
        c.content,
        c.semantic_rank,
        c.fulltext_rank,
        c.rrf_score
    FROM combined c
    ORDER BY c.rrf_score DESC
    LIMIT match_count;
END;
$$;

-- ============================================
-- VERIFICATION
-- ============================================
SELECT 'Setup completed - Table with VECTOR(1024) for voyage-3-large' AS status;
