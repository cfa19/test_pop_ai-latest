-- ============================================
-- SWITCH TEXT SEARCH FROM SPANISH TO ENGLISH
-- ============================================
-- This file replaces all text search functions and triggers
-- to use English language configuration instead of Spanish
-- ============================================

-- ============================================
-- GENERAL EMBEDDINGS TABLE FUNCTIONS
-- ============================================

-- Update trigger function for general_embeddings_1024 (ENGLISH)
CREATE OR REPLACE FUNCTION update_content_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Full-text search (keywords only) - ENGLISH
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
        ts_rank(d.content_tsvector, plainto_tsquery('english', query_text))::FLOAT AS rank
    FROM general_embeddings_1024 d
    WHERE d.content_tsvector @@ plainto_tsquery('english', query_text)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Hybrid search with RRF (Reciprocal Rank Fusion) - ENGLISH
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
                ORDER BY ts_rank(d.content_tsvector, plainto_tsquery('english', query_text)) DESC
            ) AS rank
        FROM general_embeddings_1024 d
        WHERE d.content_tsvector @@ plainto_tsquery('english', query_text)
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
-- USER EMBEDDINGS TABLE FUNCTIONS
-- ============================================

-- Update trigger function for user_embeddings_1024 (ENGLISH)
CREATE OR REPLACE FUNCTION update_user_embedding_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Full-text search for user-specific embeddings - ENGLISH
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
        ts_rank(d.content_tsvector, plainto_tsquery('english', query_text))::FLOAT AS rank
    FROM user_embeddings_1024 d
    WHERE d.user_id = filter_user_id
        AND d.content_tsvector @@ plainto_tsquery('english', query_text)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Hybrid search with RRF for user-specific embeddings - ENGLISH
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
                ORDER BY ts_rank(d.content_tsvector, plainto_tsquery('english', query_text)) DESC
            ) AS rank
        FROM user_embeddings_1024 d
        WHERE d.user_id = filter_user_id
            AND d.content_tsvector @@ plainto_tsquery('english', query_text)
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
-- UPDATE EXISTING DATA
-- ============================================

-- Regenerate tsvector for existing records in general_embeddings_1024
UPDATE general_embeddings_1024
SET content_tsvector = to_tsvector('english', content);

-- Regenerate tsvector for existing records in user_embeddings_1024
-- (Only if the table exists and has data)
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user_embeddings_1024') THEN
        UPDATE user_embeddings_1024
        SET content_tsvector = to_tsvector('english', content);
    END IF;
END $$;

-- ============================================
-- VERIFICATION
-- ============================================
SELECT 'All text search functions and triggers updated to English configuration' AS status;
