-- ============================================================
-- RUNNER CHUNK SEARCH (metadata-filtered vector similarity)
-- ============================================================
-- Returns the top-k RUN chunks whose metadata.context matches
-- ctx_key, ranked by cosine similarity to query_embedding.
-- ============================================================

CREATE OR REPLACE FUNCTION search_runner_chunks(
    query_embedding VECTOR(1024),
    ctx_key         TEXT,
    match_count     INT DEFAULT 3
)
RETURNS TABLE (
    id         BIGINT,
    content    TEXT,
    metadata   JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
    FROM general_embeddings_1024 d
    WHERE
        d.metadata->>'recommenderLabel' = 'RUN'
        AND d.metadata->>'context' = ctx_key
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
