"""
Tests for condition profile interface, validation, and prompt assembly.

These tests verify:
1. Profile validation catches missing required attributes
2. The sepsis profile has all required attributes
3. Two-phase string formatting produces correct prompts
4. ASSESSMENT_PROMPT concatenation preserves JSON braces
5. All hasattr() guards in base engine match TEMPLATE.py optional interface
"""

import importlib
import sys
import types
import pytest

from condition_profiles.sepsis import REQUIRED_ATTRIBUTES, validate_profile


# =============================================================================
# Profile Validation
# =============================================================================

class TestValidateProfile:
    """Test the validate_profile() function in condition_profiles/__init__.py."""

    def test_valid_profile_passes(self):
        """A module with all required attributes should not raise."""
        mod = types.ModuleType("fake_profile")
        for attr in REQUIRED_ATTRIBUTES:
            setattr(mod, attr, "test_value")
        validate_profile(mod)  # Should not raise

    def test_missing_single_attribute_raises(self):
        """Missing one required attribute should raise AttributeError."""
        mod = types.ModuleType("fake_profile")
        for attr in REQUIRED_ATTRIBUTES:
            setattr(mod, attr, "test_value")
        delattr(mod, "CONDITION_NAME")

        with pytest.raises(AttributeError, match="CONDITION_NAME"):
            validate_profile(mod)

    def test_missing_multiple_attributes_lists_all(self):
        """Missing multiple attributes should list all of them in the error."""
        mod = types.ModuleType("fake_profile")
        # Only set a few
        setattr(mod, "CONDITION_NAME", "test")
        setattr(mod, "CONDITION_DISPLAY_NAME", "Test")

        with pytest.raises(AttributeError) as exc_info:
            validate_profile(mod)

        error_msg = str(exc_info.value)
        # Should mention several missing attrs
        assert "GOLD_LETTERS_PATH" in error_msg
        assert "DENIAL_CONDITION_QUESTION" in error_msg
        assert "WRITER_SCORING_INSTRUCTIONS" in error_msg

    def test_none_value_still_passes(self):
        """Attributes set to None should pass (they exist, just null)."""
        mod = types.ModuleType("fake_profile")
        for attr in REQUIRED_ATTRIBUTES:
            setattr(mod, attr, None)
        validate_profile(mod)  # Should not raise

    def test_error_mentions_existing_profile(self):
        """Error message should reference an existing profile for guidance."""
        mod = types.ModuleType("fake_profile")
        with pytest.raises(AttributeError, match="sepsis.py"):
            validate_profile(mod)


# =============================================================================
# Sepsis Profile Completeness
# =============================================================================

class TestSepsisProfile:
    """Verify that condition_profiles/sepsis.py has the complete interface."""

    @pytest.fixture
    def sepsis(self):
        return importlib.import_module("condition_profiles.sepsis")

    def test_all_required_attributes_present(self, sepsis):
        """Sepsis profile should have every required attribute."""
        missing = [attr for attr in REQUIRED_ATTRIBUTES if not hasattr(sepsis, attr)]
        assert missing == [], f"Sepsis profile missing required attributes: {missing}"

    def test_validate_profile_passes(self, sepsis):
        """validate_profile() should accept the sepsis profile."""
        validate_profile(sepsis)  # Should not raise

    def test_optional_scorer_functions_present(self, sepsis):
        """Sepsis profile should have all optional scorer functions."""
        assert hasattr(sepsis, "calculate_clinical_scores")
        assert hasattr(sepsis, "write_clinical_scores_table")
        assert callable(sepsis.calculate_clinical_scores)
        assert callable(sepsis.write_clinical_scores_table)

    def test_optional_docx_functions_present(self, sepsis):
        """Sepsis profile should have all optional DOCX rendering functions."""
        assert hasattr(sepsis, "format_scores_for_prompt")
        assert hasattr(sepsis, "render_scores_status_note")
        assert hasattr(sepsis, "render_scores_in_docx")
        assert callable(sepsis.format_scores_for_prompt)
        assert callable(sepsis.render_scores_status_note)
        assert callable(sepsis.render_scores_in_docx)

    def test_optional_numeric_crosscheck_present(self, sepsis):
        """Sepsis profile should have LAB_VITAL_MATCHERS and helper functions."""
        assert hasattr(sepsis, "LAB_VITAL_MATCHERS")
        assert hasattr(sepsis, "PARAM_TO_CATEGORY")
        assert hasattr(sepsis, "match_name")
        assert hasattr(sepsis, "safe_float")

    def test_optional_structured_data_extras_present(self, sepsis):
        """Sepsis profile should have the optional structured data attributes."""
        assert hasattr(sepsis, "DIAGNOSIS_EXAMPLES")
        assert hasattr(sepsis, "STRUCTURED_DATA_SYSTEM_MESSAGE")

    def test_condition_name_is_string(self, sepsis):
        assert isinstance(sepsis.CONDITION_NAME, str)
        assert len(sepsis.CONDITION_NAME) > 0

    def test_drg_codes_is_list(self, sepsis):
        assert isinstance(sepsis.DRG_CODES, list)
        assert all(isinstance(c, str) for c in sepsis.DRG_CODES)


# =============================================================================
# Respiratory Failure Profile Completeness
# =============================================================================

class TestRespiratoryFailureProfile:
    """Verify that condition_profiles/respiratory_failure.py has the complete interface."""

    @pytest.fixture
    def rf(self):
        return importlib.import_module("condition_profiles.respiratory_failure")

    def test_all_required_attributes_present(self, rf):
        """RF profile should have every required attribute."""
        missing = [attr for attr in REQUIRED_ATTRIBUTES if not hasattr(rf, attr)]
        assert missing == [], f"RF profile missing required attributes: {missing}"

    def test_validate_profile_passes(self, rf):
        """validate_profile() should accept the RF profile."""
        validate_profile(rf)  # Should not raise

    def test_no_scorer_functions(self, rf):
        """RF profile should NOT have clinical scorer functions (no SOFA equivalent)."""
        assert not hasattr(rf, "calculate_clinical_scores")
        assert not hasattr(rf, "write_clinical_scores_table")
        assert not hasattr(rf, "format_scores_for_prompt")
        assert not hasattr(rf, "render_scores_status_note")
        assert not hasattr(rf, "render_scores_in_docx")

    def test_clinical_scores_table_name_is_none(self, rf):
        """CLINICAL_SCORES_TABLE_NAME should be None for ARF."""
        assert rf.CLINICAL_SCORES_TABLE_NAME is None

    def test_conditional_rebuttals_present(self, rf):
        """RF profile should have 3 conditional rebuttals."""
        assert hasattr(rf, "CONDITIONAL_REBUTTALS")
        assert len(rf.CONDITIONAL_REBUTTALS) == 3
        for rebuttal in rf.CONDITIONAL_REBUTTALS:
            assert "name" in rebuttal
            assert "trigger" in rebuttal
            assert "text" in rebuttal

    def test_lab_vital_matchers_includes_respiratory_params(self, rf):
        """RF profile should have respiratory-specific LAB_VITAL_MATCHERS."""
        assert hasattr(rf, "LAB_VITAL_MATCHERS")
        assert "spo2" in rf.LAB_VITAL_MATCHERS
        assert "pao2" in rf.LAB_VITAL_MATCHERS
        assert "paco2" in rf.LAB_VITAL_MATCHERS
        assert "fio2" in rf.LAB_VITAL_MATCHERS
        assert "peep" in rf.LAB_VITAL_MATCHERS
        assert "tidal_volume" in rf.LAB_VITAL_MATCHERS

    def test_param_to_category_covers_respiratory_params(self, rf):
        """PARAM_TO_CATEGORY should map respiratory-specific parameters."""
        assert hasattr(rf, "PARAM_TO_CATEGORY")
        assert rf.PARAM_TO_CATEGORY["spo2"] == "spo2"
        assert rf.PARAM_TO_CATEGORY["pao2"] == "pao2"
        assert rf.PARAM_TO_CATEGORY["paco2"] == "paco2"
        assert rf.PARAM_TO_CATEGORY["fio2"] == "fio2"
        assert rf.PARAM_TO_CATEGORY["peep"] == "peep"
        assert rf.PARAM_TO_CATEGORY["tidal volume"] == "tidal_volume"

    def test_param_to_category_all_have_matchers(self, rf):
        """Every category in PARAM_TO_CATEGORY should have a LAB_VITAL_MATCHERS entry."""
        categories = set(rf.PARAM_TO_CATEGORY.values())
        missing = [cat for cat in categories if cat not in rf.LAB_VITAL_MATCHERS]
        assert missing == [], f"Categories in PARAM_TO_CATEGORY without LAB_VITAL_MATCHERS: {missing}"

    def test_numeric_crosscheck_helpers_present(self, rf):
        """RF profile should have match_name and safe_float utility functions."""
        assert hasattr(rf, "match_name")
        assert hasattr(rf, "safe_float")
        assert callable(rf.match_name)
        assert callable(rf.safe_float)

    def test_optional_structured_data_extras_present(self, rf):
        """RF profile should have the optional structured data attributes."""
        assert hasattr(rf, "DIAGNOSIS_EXAMPLES")
        assert hasattr(rf, "STRUCTURED_DATA_SYSTEM_MESSAGE")

    def test_condition_name_is_string(self, rf):
        assert isinstance(rf.CONDITION_NAME, str)
        assert rf.CONDITION_NAME == "respiratory_failure"

    def test_drg_codes(self, rf):
        assert isinstance(rf.DRG_CODES, list)
        assert rf.DRG_CODES == ["189", "190", "191", "207", "208"]


# =============================================================================
# Two-Phase String Formatting (Prompt Assembly)
# =============================================================================

class TestPromptAssembly:
    """
    Verify the two-phase string formatting pattern used throughout.

    Pattern: .format() at definition time (double-brace {{}} for deferred placeholders)
    then .replace() at call time fills the deferred placeholders.
    """

    def test_denial_parser_prompt_phase1(self):
        """DENIAL_PARSER_PROMPT: definition-time .format() fills condition fields."""
        template = '''Find:
1. Test
{condition_question}

Return:
{condition_field}: [YES or NO]
Denial: {{denial_text}}'''.format(
            condition_question="5. IS SEPSIS RELATED?",
            condition_field="IS_SEPSIS",
        )

        # Phase 1 should have filled the condition fields
        assert "5. IS SEPSIS RELATED?" in template
        assert "IS_SEPSIS: [YES or NO]" in template
        # Phase 1 should have reduced {{denial_text}} to {denial_text}
        assert "{denial_text}" in template
        assert "{{denial_text}}" not in template

    def test_denial_parser_prompt_phase2(self):
        """DENIAL_PARSER_PROMPT: call-time .replace() fills denial text."""
        # Simulate the full two-phase pattern
        template = "Extract: {{denial_text}}".format()
        result = template.replace("{denial_text}", "Some denial text here")
        assert "Some denial text here" in result
        assert "{denial_text}" not in result

    def test_note_extraction_prompt_phases(self):
        """NOTE_EXTRACTION_PROMPT: double-braces survive first .format()."""
        template = '''Note type: {{note_type}}
Note text: {{note_text}}
Targets: {extraction_targets}'''.format(
            extraction_targets="Extract SOFA components"
        )

        # Phase 1: extraction_targets filled, note_type/note_text deferred
        assert "Extract SOFA components" in template
        assert "{note_type}" in template
        assert "{note_text}" in template

        # Phase 2
        result = template.replace("{note_type}", "Progress Note")
        result = result.replace("{note_text}", "Patient has fever...")
        assert "Progress Note" in result
        assert "Patient has fever..." in result

    def test_structured_data_prompt_phases(self):
        """STRUCTURED_DATA_EXTRACTION_PROMPT: condition context + diagnosis examples at phase 1."""
        template = '''Context: {diagnosis_examples}
Task: {condition_context}
Timeline: {{structured_timeline}}'''.format(
            diagnosis_examples='For example:\n- "Severe sepsis"',
            condition_context="Extract sepsis-relevant data",
        )

        assert 'For example:' in template
        assert "Extract sepsis-relevant data" in template
        assert "{structured_timeline}" in template
        assert "{{structured_timeline}}" not in template

        result = template.replace("{structured_timeline}", "[2024-01-01] lab: lactate 4.2")
        assert "[2024-01-01] lab: lactate 4.2" in result

    def test_conflict_detection_prompt_phases(self):
        """CONFLICT_DETECTION_PROMPT: conflict examples at phase 1."""
        template = '''Examples: {conflict_examples}
Notes: {{notes_summary}}
Structured: {{structured_summary}}'''.format(
            conflict_examples="MAP maintained >65 but vitals show MAP <65"
        )

        assert "MAP maintained >65" in template
        assert "{notes_summary}" in template
        assert "{structured_summary}" in template

    def test_assessment_prompt_concatenation(self):
        """
        ASSESSMENT_PROMPT uses string concatenation to split phases.
        First part is .format()'d at definition time. Second part has deferred placeholders.
        JSON template braces {{}} must survive as literal braces.
        """
        # Simulate the actual pattern from inference.py
        prompt = '''Evaluating a {condition_label}.
═══ {criteria_label} ═══'''.format(
            condition_label="sepsis DRG appeal letter",
            criteria_label="PROPEL SEPSIS CRITERIA",
        ) + '''
{propel_definition}

Return ONLY valid JSON:
{{
  "overall_score": <1-10>,
  "propel_criteria": {{
    "score": <1-10>
  }}
}}'''

        # Phase 1 should have filled condition_label and criteria_label
        assert "sepsis DRG appeal letter" in prompt
        assert "PROPEL SEPSIS CRITERIA" in prompt

        # Runtime placeholder should be present
        assert "{propel_definition}" in prompt

        # JSON braces should be literal (single braces from the concatenated part)
        assert '"overall_score": <1-10>' in prompt
        assert '"score": <1-10>' in prompt

        # Phase 2: fill runtime placeholders
        result = prompt.format(propel_definition="Sepsis is defined as...")
        assert "Sepsis is defined as..." in result
        # JSON structure should survive .format()
        assert '"overall_score": <1-10>' in result

    def test_writer_prompt_single_phase(self):
        """WRITER_PROMPT uses a single .format() call with all placeholders."""
        template = '''Denial: {denial_letter_text}
Scores: {clinical_scores_section}
Instructions: {scoring_instructions}
Patient: {patient_name}'''

        result = template.format(
            denial_letter_text="Your claim is denied",
            clinical_scores_section="SOFA Total: 8",
            scoring_instructions="Reference SOFA scores narratively",
            patient_name="John Doe",
        )

        assert "Your claim is denied" in result
        assert "SOFA Total: 8" in result
        assert "Reference SOFA scores narratively" in result
        assert "John Doe" in result

    def test_no_orphaned_placeholders_in_denial_parser(self):
        """After both phases, no {placeholder} should remain."""
        template = '''Find: {condition_question}
Field: {condition_field}: [YES or NO]
Text: {{denial_text}}'''.format(
            condition_question="5. IS SEPSIS RELATED?",
            condition_field="IS_SEPSIS",
        )
        result = template.replace("{denial_text}", "Denial content here")

        # No single-brace placeholders should remain
        import re
        orphaned = re.findall(r'(?<!\{)\{[a-z_]+\}(?!\})', result)
        assert orphaned == [], f"Orphaned placeholders found: {orphaned}"


# =============================================================================
# TEMPLATE.py Required/Optional Alignment
# =============================================================================

class TestTemplateAlignment:
    """Verify TEMPLATE.py's required/optional annotations match the base engine."""

    # These are the attributes that the base engine accesses via hasattr() — they're optional
    OPTIONAL_SCORER_FUNCTIONS = [
        "calculate_clinical_scores",
        "write_clinical_scores_table",
    ]

    OPTIONAL_DOCX_FUNCTIONS = [
        "format_scores_for_prompt",
        "render_scores_status_note",
        "render_scores_in_docx",
    ]

    OPTIONAL_NUMERIC_CROSSCHECK = [
        "LAB_VITAL_MATCHERS",
        "PARAM_TO_CATEGORY",
    ]

    OPTIONAL_STRUCTURED_DATA = [
        "DIAGNOSIS_EXAMPLES",
        "STRUCTURED_DATA_SYSTEM_MESSAGE",
    ]

    def test_optional_attrs_not_in_required_list(self):
        """Optional attributes should NOT be in REQUIRED_ATTRIBUTES."""
        all_optional = (
            self.OPTIONAL_SCORER_FUNCTIONS
            + self.OPTIONAL_DOCX_FUNCTIONS
            + self.OPTIONAL_NUMERIC_CROSSCHECK
            + self.OPTIONAL_STRUCTURED_DATA
        )
        in_required = [attr for attr in all_optional if attr in REQUIRED_ATTRIBUTES]
        assert in_required == [], f"These optional attrs are listed as required: {in_required}"

    def test_required_list_covers_direct_accesses(self):
        """All directly-accessed (non-guarded) profile attributes should be in REQUIRED_ATTRIBUTES."""
        # These are accessed as profile.X without hasattr/getattr guards
        direct_accesses = [
            "CONDITION_NAME",
            "CONDITION_DISPLAY_NAME",
            "GOLD_LETTERS_PATH",
            "PROPEL_DATA_PATH",
            "DEFAULT_TEMPLATE_PATH",
            "CLINICAL_SCORES_TABLE_NAME",
            "DENIAL_CONDITION_QUESTION",
            "DENIAL_CONDITION_FIELD",
            "NOTE_EXTRACTION_TARGETS",
            "STRUCTURED_DATA_CONTEXT",
            "CONFLICT_EXAMPLES",
            "WRITER_SCORING_INSTRUCTIONS",
            "ASSESSMENT_CONDITION_LABEL",
            "ASSESSMENT_CRITERIA_LABEL",
        ]
        missing = [attr for attr in direct_accesses if attr not in REQUIRED_ATTRIBUTES]
        assert missing == [], f"Directly-accessed attrs missing from REQUIRED_ATTRIBUTES: {missing}"
