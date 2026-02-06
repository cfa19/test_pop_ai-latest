-- ============================================
-- USER-SPECIFIC EMBEDDINGS TABLE
-- ============================================
-- Embedding model: voyage-3-large (1024 dimensions)
-- Provider: Voyage AI
-- Stores embeddings for user conversation messages
-- ============================================

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- TABLE FOR user-specific embeddings (1024 dims)
-- ============================================

DROP TABLE IF EXISTS user_embeddings_1024 CASCADE;

CREATE TABLE user_embeddings_1024 (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    content_tsvector TSVECTOR,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX user_embeddings_1024_user_id_idx
ON user_embeddings_1024(user_id);

CREATE INDEX user_embeddings_1024_embedding_idx
ON user_embeddings_1024
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX user_embeddings_1024_tsvector_idx
ON user_embeddings_1024
USING GIN (content_tsvector);

-- ============================================
-- TRIGGER FOR AUTOMATIC TSVECTOR (SPANISH)
-- ============================================

CREATE OR REPLACE FUNCTION update_user_embedding_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('spanish', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS user_embedding_tsvector_update ON user_embeddings_1024;
CREATE TRIGGER user_embedding_tsvector_update
    BEFORE INSERT OR UPDATE ON user_embeddings_1024
    FOR EACH ROW
    EXECUTE FUNCTION update_user_embedding_tsvector();

-- ============================================
-- ENABLE ROW LEVEL SECURITY
-- ============================================

ALTER TABLE user_embeddings_1024 ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own embeddings
CREATE POLICY "Users can view own embeddings"
    ON user_embeddings_1024
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Users can insert their own embeddings
CREATE POLICY "Users can insert own embeddings"
    ON user_embeddings_1024
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Policy: Service role has full access (for backend)
CREATE POLICY "Service role has full access to user embeddings"
    ON user_embeddings_1024
    FOR ALL
    USING (auth.role() = 'service_role');

-- ============================================
-- SEARCH FUNCTIONS FOR USER EMBEDDINGS
-- ============================================

-- Semantic search for user-specific embeddings
CREATE OR REPLACE FUNCTION user_rag_search_semantic(
    query_embedding VECTOR(1024),
    filter_user_id UUID,
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
    FROM user_embeddings_1024 d
    WHERE d.user_id = filter_user_id
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Full-text search for user-specific embeddings
CREATE OR REPLACE FUNCTION user_rag_search_fulltext(
    query_text TEXT,
    filter_user_id UUID,
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
    FROM user_embeddings_1024 d
    WHERE d.user_id = filter_user_id
        AND d.content_tsvector @@ plainto_tsquery('spanish', query_text)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Hybrid search with RRF for user-specific embeddings
CREATE OR REPLACE FUNCTION user_rag_hybrid_search(
    query_text TEXT,
    query_embedding VECTOR(1024),
    filter_user_id UUID,
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
        FROM user_embeddings_1024 d
        WHERE d.user_id = filter_user_id
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
        FROM user_embeddings_1024 d
        WHERE d.user_id = filter_user_id
            AND d.content_tsvector @@ plainto_tsquery('spanish', query_text)
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
SELECT 'User embeddings table created - VECTOR(1024) for voyage-3-large' AS status;
