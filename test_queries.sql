-- Test Queries for Sepsis Rebuttal Engine
-- Run these to validate data in Unity Catalog tables
-- Updated: Check denial data join worked properly

-- =============================================================================
-- INFERENCE TABLE VALIDATION (fudgesicle_inference)
-- =============================================================================

-- 1. Basic counts - did data load AND did denial join work?
SELECT
    COUNT(*) as total_rows,
    COUNT(discharge_summary_text) as has_discharge,
    COUNT(hp_note_text) as has_hp,
    COUNT(denial_letter_text) as has_denial_text,
    COUNT(denial_letter_filename) as has_denial_filename,
    COUNT(denial_embedding) as has_embedding,
    COUNT(payor) as has_payor
FROM dev.fin_ds.fudgesicle_inference;

-- 2. Check the actual data - KEY VALIDATION
SELECT
    hsp_account_id,
    formatted_name,
    payor,
    denial_letter_filename,
    LENGTH(denial_letter_text) as denial_chars,
    LENGTH(discharge_summary_text) as discharge_chars,
    LENGTH(hp_note_text) as hp_chars,
    SIZE(denial_embedding) as embedding_dims
FROM dev.fin_ds.fudgesicle_inference;

-- 3. Explicit NULL check - which columns have nulls?
SELECT
    hsp_account_id,
    CASE WHEN denial_letter_text IS NULL THEN 'NULL' ELSE 'OK' END as denial_text,
    CASE WHEN denial_letter_filename IS NULL THEN 'NULL' ELSE 'OK' END as denial_filename,
    CASE WHEN denial_embedding IS NULL THEN 'NULL' ELSE 'OK' END as embedding,
    CASE WHEN payor IS NULL THEN 'NULL' ELSE 'OK' END as payor,
    CASE WHEN discharge_summary_text IS NULL OR discharge_summary_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as discharge,
    CASE WHEN hp_note_text IS NULL OR hp_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as hp
FROM dev.fin_ds.fudgesicle_inference;

-- 4. Preview note content
SELECT
    hsp_account_id,
    LEFT(discharge_summary_text, 200) as discharge_preview,
    LEFT(hp_note_text, 200) as hp_preview
FROM dev.fin_ds.fudgesicle_inference
LIMIT 3;

-- 5. Check embedding exists
SELECT
    hsp_account_id,
    denial_embedding[0] as first_dim
FROM dev.fin_ds.fudgesicle_inference
WHERE denial_embedding IS NOT NULL;


-- =============================================================================
-- GOLD LETTERS TABLE VALIDATION (fudgesicle_gold_letters)
-- =============================================================================

-- 6. Check gold letters loaded correctly
SELECT
    source_file,
    payor,
    LENGTH(rebuttal_text) as rebuttal_chars,
    LENGTH(denial_text) as denial_chars,
    metadata['denial_start_page'] as denial_start_page,
    metadata['total_pages'] as total_pages,
    SIZE(denial_embedding) as embedding_dims
FROM dev.fin_ds.fudgesicle_gold_letters;

-- 7. Preview gold letter text (rebuttal vs denial split)
SELECT
    source_file,
    LEFT(rebuttal_text, 200) as rebuttal_preview,
    LEFT(denial_text, 200) as denial_preview
FROM dev.fin_ds.fudgesicle_gold_letters
LIMIT 3;

-- 8. Verify gold letter embeddings
SELECT
    source_file,
    denial_embedding[0] as first_val,
    denial_embedding[1535] as last_val
FROM dev.fin_ds.fudgesicle_gold_letters;


-- =============================================================================
-- CROSS-TABLE CHECKS
-- =============================================================================

-- 9. Compare payor distribution
SELECT 'inference' as source, payor, COUNT(*) as count
FROM dev.fin_ds.fudgesicle_inference
GROUP BY payor
UNION ALL
SELECT 'gold_letters' as source, payor, COUNT(*) as count
FROM dev.fin_ds.fudgesicle_gold_letters
GROUP BY payor
ORDER BY source, count DESC;

-- 10. Total record counts
SELECT
    (SELECT COUNT(*) FROM dev.fin_ds.fudgesicle_gold_letters) as gold_letters,
    (SELECT COUNT(*) FROM dev.fin_ds.fudgesicle_inference) as inference_rows;
