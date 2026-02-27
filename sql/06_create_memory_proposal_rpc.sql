-- ============================================================================
-- RPC: create_memory_proposal
-- Inserts a memory card with DB-level consent check + type validation.
-- Third layer of defense (after orchestrator + Python validation).
-- ============================================================================

CREATE OR REPLACE FUNCTION create_memory_proposal(
  p_user_id UUID,
  p_content TEXT,
  p_type TEXT,
  p_confidence NUMERIC,
  p_source JSONB,
  p_raw_data JSONB DEFAULT NULL,
  p_tags TEXT[] DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
  v_consent BOOLEAN;
  v_card_id UUID;
BEGIN
  -- DB-level consent check (Layer 3 — defense in depth)
  SELECT consent_given INTO v_consent
  FROM user_consents
  WHERE user_id = p_user_id
    AND consent_type = 'ai_training'
    AND consent_given = true
    AND revoked_at IS NULL
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_consent IS NOT TRUE THEN
    RAISE EXCEPTION 'No ai_training consent for user %', p_user_id
      USING ERRCODE = 'P0001';
  END IF;

  -- Type validation is enforced by CHECK constraint on memory_cards.type column
  -- Status set to 'validated' — cards are automatically active
  INSERT INTO memory_cards (user_id, content, type, confidence, source, raw_data, tags, status)
  VALUES (p_user_id, p_content, p_type, p_confidence, p_source, p_raw_data, p_tags, 'validated')
  RETURNING id INTO v_card_id;

  RETURN v_card_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute to service role only
REVOKE ALL ON FUNCTION create_memory_proposal FROM PUBLIC;
GRANT EXECUTE ON FUNCTION create_memory_proposal TO service_role;
