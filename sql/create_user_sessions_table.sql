-- User Sessions Table
-- Tracks active user sessions and last activity

CREATE TABLE IF NOT EXISTS coach_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    email TEXT,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT coach_sessions_user_id_key UNIQUE (user_id)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_coach_sessions_user_id ON coach_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_coach_sessions_last_seen ON coach_sessions(last_seen DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE coach_sessions ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own session
CREATE POLICY "Users can view own session"
    ON coach_sessions
    FOR SELECT
    USING (auth.uid() = user_id);

-- Policy: Service role can do anything (for backend)
CREATE POLICY "Service role has full access"
    ON coach_sessions
    FOR ALL
    USING (auth.role() = 'service_role');

-- Function to clean up old sessions (optional, run periodically)
CREATE OR REPLACE FUNCTION cleanup_old_sessions(inactive_minutes INTEGER DEFAULT 60)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM coach_sessions
    WHERE last_seen < NOW() - (inactive_minutes || ' minutes')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get online users count
CREATE OR REPLACE FUNCTION get_online_users_count(active_minutes INTEGER DEFAULT 5)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM coach_sessions
        WHERE last_seen > NOW() - (active_minutes || ' minutes')::INTERVAL
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON TABLE coach_sessions IS 'Tracks active user sessions and last activity timestamps';
COMMENT ON COLUMN coach_sessions.user_id IS 'User ID from Supabase auth';
COMMENT ON COLUMN coach_sessions.last_seen IS 'Last activity timestamp';
COMMENT ON COLUMN coach_sessions.metadata IS 'Optional: store user agent, IP, device info, etc.';
