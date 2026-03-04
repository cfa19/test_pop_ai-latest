-- ============================================================================
-- Memory Extraction Queue
-- Reliable queue for processing runner completions into memory cards.
-- Uses PostgreSQL trigger + FOR UPDATE SKIP LOCKED for exactly-once processing.
-- ============================================================================

-- Queue table
CREATE TABLE IF NOT EXISTS memory_extraction_queue (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  completion_id UUID NOT NULL REFERENCES activity_completions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  CONSTRAINT valid_queue_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
  CONSTRAINT unique_completion UNIQUE (completion_id)
);

-- Index for queue polling: pending items ordered by creation time
CREATE INDEX IF NOT EXISTS idx_extraction_queue_pending
  ON memory_extraction_queue (created_at)
  WHERE status = 'pending';

-- Index for failed items eligible for retry
CREATE INDEX IF NOT EXISTS idx_extraction_queue_retry
  ON memory_extraction_queue (created_at)
  WHERE status = 'failed' AND attempts < 3;

-- Trigger: auto-enqueue when a completion reaches status='completed'
CREATE OR REPLACE FUNCTION enqueue_memory_extraction()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.status = 'completed' AND (TG_OP = 'INSERT' OR OLD.status IS DISTINCT FROM 'completed') THEN
    INSERT INTO memory_extraction_queue (completion_id, user_id)
    VALUES (NEW.id, NEW.user_id)
    ON CONFLICT (completion_id) DO NOTHING;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if any, then create
DROP TRIGGER IF EXISTS trg_enqueue_memory_extraction ON activity_completions;
CREATE TRIGGER trg_enqueue_memory_extraction
  AFTER INSERT OR UPDATE ON activity_completions
  FOR EACH ROW EXECUTE FUNCTION enqueue_memory_extraction();

-- Webhook trigger: notify Python API when a new item is enqueued
-- pg_net calls POST /api/extract so the QueueConsumer wakes immediately
CREATE OR REPLACE FUNCTION notify_extraction_webhook()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM net.http_post(
    url := 'http://127.0.0.1:8000/api/extract',
    body := '{}'::jsonb,
    headers := '{"Content-Type": "application/json"}'::jsonb
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notify_extraction_webhook ON memory_extraction_queue;
CREATE TRIGGER trg_notify_extraction_webhook
  AFTER INSERT ON memory_extraction_queue
  FOR EACH ROW EXECUTE FUNCTION notify_extraction_webhook();

-- RPC: claim a batch of pending items for processing
-- Uses FOR UPDATE SKIP LOCKED to prevent concurrent processing
-- Also recovers items stuck in 'processing' for >10 minutes (crash recovery)
CREATE OR REPLACE FUNCTION claim_extraction_batch(batch_size INTEGER DEFAULT 10, max_retries INTEGER DEFAULT 3)
RETURNS SETOF memory_extraction_queue AS $$
BEGIN
  RETURN QUERY
  UPDATE memory_extraction_queue
  SET status = 'processing', started_at = now(), attempts = attempts + 1
  WHERE id IN (
    SELECT id FROM memory_extraction_queue
    WHERE status = 'pending'
       OR (status = 'failed' AND attempts < max_retries)
       OR (status = 'processing' AND started_at < now() - interval '10 minutes')
    ORDER BY created_at
    LIMIT batch_size
    FOR UPDATE SKIP LOCKED
  )
  RETURNING *;
END;
$$ LANGUAGE plpgsql;
