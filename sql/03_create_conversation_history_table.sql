-- Conversation History Table with Vector Embeddings
-- Stores all conversation messages with embeddings for RAG-based retrieval
-- Embedding model: voyage-3-large (1024 dimensions) via Voyage AI

-- Drop existing objects if they exist
DROP TABLE IF EXISTS conversation_history CASCADE;

CREATE TABLE conversation_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    message TEXT NOT NULL,
    embedding_id BIGINT REFERENCES user_embeddings_1024(id) ON DELETE SET NULL, -- Optional foreign key to user embeddings
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for better performance
DROP INDEX IF EXISTS idx_conversation_history_conversation_id;
DROP INDEX IF EXISTS idx_conversation_history_user_id;
DROP INDEX IF EXISTS idx_conversation_history_created_at;
DROP INDEX IF EXISTS idx_conversation_history_embedding_id;

CREATE INDEX idx_conversation_history_conversation_id ON conversation_history(conversation_id);
CREATE INDEX idx_conversation_history_user_id ON conversation_history(user_id);
CREATE INDEX idx_conversation_history_created_at ON conversation_history(created_at DESC);
CREATE INDEX idx_conversation_history_embedding_id ON conversation_history(embedding_id);

-- Enable Row Level Security (RLS)
ALTER TABLE conversation_history ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own conversation history
CREATE POLICY "Users can view own conversations"
    ON conversation_history
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Service role has full access (for backend)
CREATE POLICY "Service role has full access"
    ON conversation_history
    FOR ALL
    USING (auth.role() = 'service_role');

-- Function to search conversation history using vector similarity
-- Joins with user_embeddings_1024 table to get embeddings
CREATE OR REPLACE FUNCTION search_conversation_history(
    query_embedding vector(1024),
    filter_conversation_id TEXT,
    filter_user_id UUID,
    match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    conversation_id TEXT,
    role TEXT,
    message TEXT,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ch.id,
        ch.conversation_id,
        ch.role,
        ch.message,
        1 - (ue.embedding <=> query_embedding) AS similarity,
        ch.created_at
    FROM conversation_history ch
    INNER JOIN user_embeddings_1024 ue ON ch.embedding_id = ue.id
    WHERE ch.conversation_id = filter_conversation_id
        AND ch.user_id = filter_user_id
        AND ch.embedding_id IS NOT NULL
    ORDER BY ue.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clean up old conversations (optional)
CREATE OR REPLACE FUNCTION cleanup_old_conversations(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM conversation_history
    WHERE created_at < NOW() - (days_old || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON TABLE conversation_history IS 'Stores conversation messages with optional references to user embeddings for RAG-based retrieval';
COMMENT ON COLUMN conversation_history.conversation_id IS 'Unique identifier for the conversation session';
COMMENT ON COLUMN conversation_history.role IS 'Message role: user or assistant';
COMMENT ON COLUMN conversation_history.embedding_id IS 'Optional foreign key to user_embeddings_1024 table; only worthy messages have embeddings';
