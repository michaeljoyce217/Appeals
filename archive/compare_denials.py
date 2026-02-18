# compare_denials.py
# Compare Sample_Denial_Letters with denial portions from gold_standard_appeals
# Run in Databricks to check if they contain the same content

# =============================================================================
# CELL 1: Setup (copy paths from featurization.py)
# =============================================================================
GOLD_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/gold_standard_appeals.gold_standard_appeals__sepsis_only"
DENIAL_LETTERS_PATH = "/Workspace/Repos/mijo8881@mercy.net/fudgesicle/utils/sample_denial_letters"

# =============================================================================
# CELL 2: Import and setup Document Intelligence
# =============================================================================
import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

AZURE_DOC_INTEL_KEY = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-key1')
AZURE_DOC_INTEL_ENDPOINT = dbutils.secrets.get(scope='idp_etl', key='az-aidcmntintel-endpoint')

doc_intel_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTEL_KEY)
)

def extract_text_from_pdf(file_path):
    """Extract text from PDF using Document Intelligence."""
    with open(file_path, 'rb') as f:
        document_bytes = f.read()
    poller = doc_intel_client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=document_bytes),
    )
    result = poller.result()
    pages_text = []
    for page in result.pages:
        page_lines = [line.content for line in page.lines]
        pages_text.append("\n".join(page_lines))
    return pages_text

print("Setup complete")

# =============================================================================
# CELL 3: Extract sample denial texts
# =============================================================================
print("="*60)
print("EXTRACTING SAMPLE DENIAL LETTERS")
print("="*60)

sample_denials = {}
sample_files = [f for f in os.listdir(DENIAL_LETTERS_PATH) if f.lower().endswith('.pdf')]

for i, filename in enumerate(sample_files):
    print(f"[{i+1}/{len(sample_files)}] {filename}")
    file_path = os.path.join(DENIAL_LETTERS_PATH, filename)
    try:
        pages = extract_text_from_pdf(file_path)
        full_text = "\n\n".join(pages)
        sample_denials[filename] = full_text
        print(f"  {len(full_text)} chars")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nExtracted {len(sample_denials)} sample denials")

# =============================================================================
# CELL 4: Extract denial portions from gold letters
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING DENIAL PORTIONS FROM GOLD LETTERS")
print("="*60)

# Insurance indicators for finding denial start
insurance_indicators = [
    "unitedhealth", "aetna", "cigna", "humana", "anthem",
    "blue cross", "blue shield", "bcbs", "kaiser", "molina"
]

def find_denial_start(pages_text):
    """Find which page the denial starts on (skip page 1)."""
    for i, page_text in enumerate(pages_text):
        if i == 0:
            continue  # Skip first page (rebuttal)
        header = "\n".join(page_text.split("\n")[:10]).lower()
        for indicator in insurance_indicators:
            if indicator in header:
                return i
    return len(pages_text)  # Default: no denial found

gold_denials = {}
gold_files = [f for f in os.listdir(GOLD_LETTERS_PATH) if f.lower().endswith('.pdf')]

for i, filename in enumerate(gold_files):
    print(f"[{i+1}/{len(gold_files)}] {filename}")
    file_path = os.path.join(GOLD_LETTERS_PATH, filename)
    try:
        pages = extract_text_from_pdf(file_path)
        denial_start = find_denial_start(pages)
        denial_pages = pages[denial_start:]
        denial_text = "\n\n".join(denial_pages)
        gold_denials[filename] = denial_text
        print(f"  Denial starts page {denial_start + 1}, {len(denial_text)} chars")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nExtracted {len(gold_denials)} gold denial portions")

# =============================================================================
# CELL 5: Compare texts - check for matches
# =============================================================================
print("\n" + "="*60)
print("COMPARING SAMPLE DENIALS TO GOLD DENIAL PORTIONS")
print("="*60)

def similarity_score(text1, text2):
    """Simple similarity: ratio of matching characters."""
    if not text1 or not text2:
        return 0.0
    # Normalize whitespace
    t1 = " ".join(text1.split()).lower()
    t2 = " ".join(text2.split()).lower()
    # Check if one contains the other
    if t1 in t2 or t2 in t1:
        return 1.0
    # Character overlap
    shorter = min(len(t1), len(t2))
    matches = sum(1 for i in range(shorter) if t1[i] == t2[i])
    return matches / shorter if shorter > 0 else 0.0

def first_n_chars_match(text1, text2, n=500):
    """Check if first N chars match (ignoring whitespace)."""
    t1 = " ".join(text1.split()).lower()[:n]
    t2 = " ".join(text2.split()).lower()[:n]
    return t1 == t2

matches_found = []

for sample_file, sample_text in sample_denials.items():
    print(f"\n{sample_file}:")

    for gold_file, gold_text in gold_denials.items():
        # Check first 500 chars
        if first_n_chars_match(sample_text, gold_text, 500):
            print(f"  EXACT MATCH (first 500 chars): {gold_file}")
            matches_found.append((sample_file, gold_file, "EXACT"))
        elif first_n_chars_match(sample_text, gold_text, 200):
            print(f"  PARTIAL MATCH (first 200 chars): {gold_file}")
            matches_found.append((sample_file, gold_file, "PARTIAL"))

# =============================================================================
# CELL 6: Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if matches_found:
    print(f"\nFOUND {len(matches_found)} MATCHES:")
    for sample, gold, match_type in matches_found:
        print(f"  {sample}")
        print(f"    -> {gold} ({match_type})")
    print("\n*** Sample denials contain the SAME content as gold letter denials ***")
    print("*** For a real POC, you need NEW denial letters ***")
else:
    print("\nNo matches found - sample denials appear to be different from gold letters.")
    print("The 1.000 similarity scores may be due to very similar denial language from the same payor.")
