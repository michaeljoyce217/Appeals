-- Test Queries for Sepsis Appeal Engine
-- Run these to validate data in Unity Catalog tables
-- Updated: Check denial data join worked properly

-- =============================================================================
-- INFERENCE TABLE VALIDATION (fudgesicle_inference)
-- =============================================================================

-- 1. Basic counts - did data load AND did denial join work?
-- All 14 sepsis-relevant note types
SELECT
    COUNT(*) as total_rows,
    -- Core notes
    COUNT(discharge_summary_text) as has_discharge,
    COUNT(hp_note_text) as has_hp,
    COUNT(progress_note_text) as has_progress,
    COUNT(consult_note_text) as has_consult,
    -- ED notes (3 types)
    COUNT(ed_notes_text) as has_ed_notes,
    COUNT(ed_triage_text) as has_ed_triage,
    COUNT(ed_provider_note_text) as has_ed_provider,
    -- Other clinical notes
    COUNT(initial_assessment_text) as has_initial_assessment,
    COUNT(addendum_note_text) as has_addendum,
    COUNT(hospital_course_text) as has_hospital_course,
    COUNT(subjective_objective_text) as has_subjective_objective,
    COUNT(assessment_plan_text) as has_assessment,
    COUNT(nursing_note_text) as has_nursing,
    COUNT(code_documentation_text) as has_code_doc,
    -- Denial data
    COUNT(denial_letter_text) as has_denial_text,
    COUNT(denial_letter_filename) as has_denial_filename,
    COUNT(denial_embedding) as has_embedding,
    COUNT(payor) as has_payor,
    COUNT(original_drg) as has_original_drg,
    COUNT(proposed_drg) as has_proposed_drg,
    SUM(CASE WHEN is_sepsis THEN 1 ELSE 0 END) as sepsis_count
FROM dev.fin_ds.fudgesicle_inference;

-- 2. Check the actual data - KEY VALIDATION (note lengths)
SELECT
    hsp_account_id,
    formatted_name,
    payor,
    original_drg,
    proposed_drg,
    is_sepsis,
    denial_letter_filename,
    LENGTH(denial_letter_text) as denial_chars,
    -- All 14 note types
    LENGTH(discharge_summary_text) as discharge_chars,
    LENGTH(hp_note_text) as hp_chars,
    LENGTH(progress_note_text) as progress_chars,
    LENGTH(consult_note_text) as consult_chars,
    LENGTH(ed_notes_text) as ed_notes_chars,
    LENGTH(initial_assessment_text) as initial_assessment_chars,
    LENGTH(ed_triage_text) as ed_triage_chars,
    LENGTH(ed_provider_note_text) as ed_provider_chars,
    LENGTH(addendum_note_text) as addendum_chars,
    LENGTH(hospital_course_text) as hospital_course_chars,
    LENGTH(subjective_objective_text) as subjective_objective_chars,
    LENGTH(assessment_plan_text) as assessment_chars,
    LENGTH(nursing_note_text) as nursing_chars,
    LENGTH(code_documentation_text) as code_doc_chars,
    SIZE(denial_embedding) as embedding_dims
FROM dev.fin_ds.fudgesicle_inference;

-- 3. Explicit NULL check - which columns have nulls? (all 14 note types)
SELECT
    hsp_account_id,
    CASE WHEN denial_letter_text IS NULL THEN 'NULL' ELSE 'OK' END as denial_text,
    CASE WHEN denial_letter_filename IS NULL THEN 'NULL' ELSE 'OK' END as denial_filename,
    CASE WHEN denial_embedding IS NULL THEN 'NULL' ELSE 'OK' END as embedding,
    CASE WHEN payor IS NULL THEN 'NULL' ELSE 'OK' END as payor,
    -- 14 note types
    CASE WHEN discharge_summary_text IS NULL OR discharge_summary_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as discharge,
    CASE WHEN hp_note_text IS NULL OR hp_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as hp,
    CASE WHEN progress_note_text IS NULL OR progress_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as progress,
    CASE WHEN consult_note_text IS NULL OR consult_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as consult,
    CASE WHEN ed_notes_text IS NULL OR ed_notes_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as ed_notes,
    CASE WHEN initial_assessment_text IS NULL OR initial_assessment_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as initial_assess,
    CASE WHEN ed_triage_text IS NULL OR ed_triage_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as ed_triage,
    CASE WHEN ed_provider_note_text IS NULL OR ed_provider_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as ed_provider,
    CASE WHEN addendum_note_text IS NULL OR addendum_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as addendum,
    CASE WHEN hospital_course_text IS NULL OR hospital_course_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as hosp_course,
    CASE WHEN subjective_objective_text IS NULL OR subjective_objective_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as subj_obj,
    CASE WHEN assessment_plan_text IS NULL OR assessment_plan_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as assessment,
    CASE WHEN nursing_note_text IS NULL OR nursing_note_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as nursing,
    CASE WHEN code_documentation_text IS NULL OR code_documentation_text = 'No Note Available' THEN 'MISSING' ELSE 'OK' END as code_doc
FROM dev.fin_ds.fudgesicle_inference;

-- 4. Preview note content (core notes)
SELECT
    hsp_account_id,
    LEFT(discharge_summary_text, 200) as discharge_preview,
    LEFT(hp_note_text, 200) as hp_preview
FROM dev.fin_ds.fudgesicle_inference
LIMIT 3;

-- 4b. Preview additional sepsis-relevant notes
SELECT
    hsp_account_id,
    LEFT(progress_note_text, 150) as progress_preview,
    LEFT(consult_note_text, 150) as consult_preview,
    LEFT(ed_note_text, 150) as ed_preview,
    LEFT(assessment_plan_text, 150) as assessment_preview,
    LEFT(hospital_course_text, 150) as hospital_course_preview
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

-- 7. Preview gold letter text (appeal vs denial split)
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
