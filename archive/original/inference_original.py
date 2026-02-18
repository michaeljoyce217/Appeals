#---------------------
# Imports &amp; settings
# ---------------------

# !pip install openai==1.88.0
# !pip install tiktoken==0.08.0
# %restart_python

from openai import AzureOpenAI
import pandas as pd
import os
import tiktoken
import datetime
from utils.helper import get_project_tbls, transfer_ownership
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp

# -----------------
# Set the catalog
# -----------------

trgt_cat = os.environ.get('trgt_cat')
spark.sql('use catalog prod;') if trgt_cat == 'dev' else spark.sql(f'use catalog {trgt_cat};')

# --------------------------------------------------------
# Check if there are new patient records to process
# --------------------------------------------------------

last_processed_ts = spark.sql(f"""
    select coalesce(max(insert_tsp), TIMESTAMP'2025-07-01 00:00:00') as last_ts
    from {trgt_cat}.fin_ds.popsicle_inference_score
""").collect()[0]["last_ts"]

# count how many new rows since last date
n_new_rows = spark.sql(f"""
    with scored_accounts as (
        select distinct hsp_account_id
        from {trgt_cat}.fin_ds.popsicle_inference_score
    )
    select count(*) as cnt
    from {trgt_cat}.fin_ds.popsicle_inference src
    left join scored_accounts sa
        on src.hsp_account_id = sa.hsp_account_id
    where src.insert_tsp > TIMESTAMP'{last_processed_ts}'
       or sa.hsp_account_id is null   -- new accounts not yet generated
""").collect()[0]["cnt"]

if n_new_rows > 0:
    print(f"{n_new_rows} new rows found, processing...")

# --------------------
# Pull medicare data
# --------------------

    df_medicare = spark.sql(f"""
    with scored_accounts as (
        select distinct hsp_account_id
        from {trgt_cat}.fin_ds.popsicle_inference_score
    )
    select src.*
    from {trgt_cat}.fin_ds.popsicle_inference src
    left join scored_accounts sa
        on src.hsp_account_id = sa.hsp_account_id
    where src.letter_type = 'Medicare'
      and (
           src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           or sa.hsp_account_id is null   -- new accounts not yet generated
      )
    """).toPandas()

    if df_medicare.empty:
        print("No unprocessed Medicare rows found.")
    else:
        print(f"{len(df_medicare)} Medicare rows to process.")

    # -----------
    # Creds
    # -----------

    api_key = "DUMMY_AZURE_OPENAI_KEY"
    azure_endpoint = "DUMMY_AZURE_OPENAI_ENDPOINT"
    api_version = '2024-10-21'
    model = 'gpt-4.1'
    client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version = api_version)

    os.environ["TIKTOKEN_CACHE_DIR"] = "" # Optional cache setting
    enc = tiktoken.encoding_for_model('gpt-4o')

    # ------------------------
    # insert medicare prompt
    # ------------------------

    medicare_prompt = """
    # Role
    You are a Medicare compliance specialist with expert knowledge of CMS regulations, inpatient medical necessity criteria, and the Two-Midnight Rule, also known as as the two-day rule.

    # Task
    Complete a Medicare medical necessity appeal letter template using the provided medical records. The patient’s claim was denied for an alleged violation of the Two-Midnight Rule (i.e., the admission lacked a reasonable expectation of requiring hospital care spanning two or more midnights).

    **Input:** Medical record data. Appeal letter template containing bracketed fields to be filled (e.g., [reason_for_admission], [summary_of_care])
    **Output:** A completed final appeal letter with all bracketed fields filled using the provided data. Return only the completed letter in plain text (no JSON).

    # Patient dischage summary text
    - Use the following text as the patient discharge summary:
    {discharge_summary}

    # HP Summary text
    - Use the folling text as the patient HP summary:
    {hp_summary}

    **Key Requirements:**
    1. If any information is unavailable, set its value to null. If notes are missing in [hp_summary], say "no H&amp;P notes available"
    2. For summary_of_care:
        - Focus on what justified inpatient admission at the time of decision
        - Include specific clinical indicators (e.g., unstable vitals, comorbidities, abnormal labs)
        - Use bullet points for clarity
    3. Emphasize specific clinical factors, patient acuity, and medical decision-making that justify inpatient admission per CMS policy.
    4. Include lab results from [hp_summary] that are relevant to the Two-Midnight Rule and justification for the admission.
    5. If a statement similar to:
        - "This patient was admitted under inpatient status. I anticipate that the patient will be hospitalized for greater than 48 hours based on presentation" appears in [hp_summary], insert it quoted above the “Therefore, Mercy Hospital is contesting...” sentence. Precede it with: When deciding to admit your member, the provider reasonably expected that hospital care would span at least two midnights, as indicated by this statement:
    6. Do not repeat the reasonable expectation statement as a bullet point in summary_of_care. Instead, extract its core message and incorporate it into the closing paragraph.
    7. Use {formatted_name} on summary of care instead of "patient"

    Fill these keys:
    - add in {mapped_address} at the beginning of the letter
    - patient_name: Exactly "{formatted_name}"
    - reason_for_admission: Extracted or summarized.
    - format "{total_charges}" in the actual dollar amount i.e if 15300.88 make it 15,300.88
    - hospital_course: use the exact verbatim of the 'Hospital Course' section within the [discharge_summary] only include the section that is labeled "HOSPITAL COURSE:" Please do not use sentence that includes "see “H and P” would not be needed" when appropriate, look in [hp_summary] for supporting lab results and vitals.
    - summary_of_care: Bullet points justifying 2-midnight rule within the [discharge_summary]
    - number_of_midnights: {num_midnights}
    - facility name: exclude the word "Parent" in {facility_name}
    - Dates of Service: exactly {formatted_date_of_service}
    - Requested for Certification: {formatted_date_of_service_ip}
    - Claim #: {claim_number}
    - if {day2_ip_date} is used please use format MM/DD/YYYY

    # Guidelines
    - Use precise medical and legal terminology suitable for Medicare Administrative Contractors (MACs) and CMS reviewers.
    - Prioritize clinical justification over administrative convenience
    - Reference specific CMS guidelines and medical criteria when applicable
    - Maintain professional, objective tone throughout
    - Ensure the letter is concise, clear
    - Only use word "observation" only in the [treatment_provided_text], use monitoring instead if needed
    - ONLY If the statement similar to "This patient was admitted under inpatient status. I anticipate that the patient will be hospitalized for greater than 48 hours based on presentation" exists in the [hp_summary] then plug it above the treatment provided section the "Therefore, Mercy Hospital is contesting the denial of coverage for your member's inpatient admission on behalf of your member, {formatted_name}" sentence. Put it in quotes
    - Make sure to only include results from the [hp_summary] that are related to the [formatted_date_of_service] dates

    # Response Format
    Return the final letter text, not json
    For the summary of care, use bullet points to clearly document the clinical rationale supporting medical necessity, make sure you are thorough and specific.

    #Template
    {mapped_address}

    Member Name: {formatted_name}
    DOB: {formatted_birthdate}
    Policy #: {policy_number}
    Claim #: {claim_number}
    Total Charges: $ {total_charges}
    Hospital Account #: {hsp_account_id}
    Dates of Service: {formatted_date_of_service}
    Requested for Certification: {formatted_date_of_service_ip}
    Facility Name: {facility_name}
    NPI: {npi}
    Tax ID: {tax_id}

    Dear Medical Director Reviewer:

    Mercy Hospital is appealing the denial of inpatient services for the above member’s hospitalization. {formatted_name} met CMS's criteria for inpatient admission based on the 2-midnight rule as clarified in the effectuated regulatory guidance within CMS 4201-F. We request a reversal of the denial and inpatient payment for this claim.

    Regulatory guidance under CMS-4201-F, effective June 5, 2023, clarified that Medicare Advantage plans must comply with general coverage and benefit conditions included in Traditional Medicare regulations and that Medicare Advantage Plans must adhere to Traditional Medicare coverage criteria, the Two-Midnight-Rule, and the Inpatient Only List (IPO) to be compliant with this regulation.

    CMS-4201-F page 223-224 states: "In regards to inpatient admissions at 412.3, we confirm that the criteria listed at 412.3(a)-(d) apply to MA. We acknowledge that 412.3 is a payment rule for Medicare FFS, however, providing payment for an item or service is one way that MA organizations provide coverage for benefits. Therefore, under § 422.101(b)(2), an MA plan must provide coverage, by furnishing, arranging for, or paying for an inpatient admission when, based on consideration of complex medical factors documented in the medical record, the admitting physician expects the patient to require hospital care that crosses two-midnights (§ 412.3(d)(1), the “two midnight benchmark”)….” This regulation prohibits MA organizations from using internal coverage criteria that alter the payment coverage criteria already established under Traditional Medicare regulations.

    Therefore, Mercy Hospital is contesting the denial of coverage for your member's inpatient admission on behalf of your member, {formatted_name}.

    {treatment_provided_text}

    Hospital Course: [hospital_course] [hp_summary to help support the hospital course]

    Summary of Care: [summary_of_care] + any supported lab results or vitals rom [hp_summary]

    In closing, {formatted_name} received medically necessary Inpatient services [use the last part of of the summary of care here that describes their medically necessary expectation of inpatient services] that spanned {num_midnights}. Mercy Hospital has established that the patient met the definition of an inpatient as per CMS guidelines and we request Inpatient claim payment following review.

    We look forward to your prompt response. Please forward correspondence to the attention of:
    Mercy Hospital
    Payor Audits &amp; Denials Department
    Attn: Compliance Manager
    2115 S Fremont Ave - Ste LL1
    Springfield, MO 65804
    RightFax: 314-364-6231


    Respectfully,
    """

    def generate_letter_text(row):
        # Extract fields from the row
        mapped_address = row.get('mapped_address')
        formatted_name = row.get('formatted_name')
        formatted_birthdate = row.get('formatted_birthdate')
        hsp_account_id = row.get('hsp_account_id')
        policy_number = row.get('policy_number')
        num_midnights = row.get('number_of_midnights')
        discharge_summary = row.get('discharge_summary_text')
        hp_summary = row.get('HP_note_text')
        facility_name = row.get('facility_name')
        total_charges = row.get('total_charges')
        formatted_date_of_service = row.get('formatted_date_of_service')
        formatted_date_of_service_ip = row.get('formatted_date_of_service_ip')
        claim_number = row.get('claim_number')
        tax_id = row.get('tax_id')
        npi = row.get('npi')
        op_ip_flag = row.get('op_ip_flag')
        day2_ip_date = row.get('day2_ip_date')

        if op_ip_flag == "yes":
            treatment_provided_text = f"""
            Clinical Presentation: Upon emergent arrival at Mercy Hospital, {formatted_name} was originally placed in observation services for brief [reason_for_admission]. On {day2_ip_date}, {formatted_name} was admitted as an Inpatient due to inability to be discharged after one midnight as {formatted_name} required continued hospital care.
            """
            logit_bias = None #adding new logic to include word observation

        else:
            treatment_provided_text = f"""
            Clinical Presentation: Upon emergent arrival at Mercy Hospital, {formatted_name} was admitted as an inpatient for [reason_for_admission] + any supported lab results or vitals from [hp_summary]. The admitting Provider had a reasonable expectation of a 2 midnight stay due to
            """
            obs_tokens = enc.encode('observation')
            obs_tokens += enc.encode('Observation')
            logit_bias = {str(token_id): -100 for token_id in obs_tokens}

        formatted_prompt_medicare =  medicare_prompt.format(
                mapped_address=mapped_address,
                formatted_name=formatted_name,
                formatted_birthdate=formatted_birthdate,
                hsp_account_id=hsp_account_id,
                policy_number=policy_number,
                num_midnights=num_midnights,
                discharge_summary=discharge_summary,
                hp_summary=hp_summary,
                facility_name=facility_name,
                total_charges=total_charges,
                formatted_date_of_service=formatted_date_of_service,
                formatted_date_of_service_ip=formatted_date_of_service_ip,
                claim_number=claim_number,
                tax_id=tax_id,
                npi=npi,
                day2_ip_date=day2_ip_date,
                treatment_provided_text=treatment_provided_text
                )

        messages_medicare = [
             {"role": "system", "content": formatted_prompt_medicare}
        ]

        # Make the API call
        response_object = client.chat.completions.create(
            messages=messages_medicare,
            model=model,
            temperature=0,
            max_tokens=5000,
            logit_bias=logit_bias
        )

        # Extract and return the response text
        return response_object.choices[0].message.content

    # Apply function and store result in new column
    # df_medicare = df_medicare.sample(15)
    df_medicare['letter_text'] = df_medicare.apply(generate_letter_text, axis=1)
    df_medicare['letter_curated_date'] = pd.to_datetime("today").normalize()
    df_medicare['letter_type'] = 'medicare'

    # --------------------
    # Pull commercial data
    # --------------------

    df_commercial = spark.sql(f"""
    with scored_accounts as (
        select distinct hsp_account_id
        from {trgt_cat}.fin_ds.popsicle_inference_score
    )
    select src.*
    from {trgt_cat}.fin_ds.popsicle_inference src
    left join scored_accounts sa
        on src.hsp_account_id = sa.hsp_account_id
    where src.letter_type = 'Commercial'
        and (
           src.insert_tsp > TIMESTAMP'{last_processed_ts}'
           or sa.hsp_account_id is null   -- new accounts not yet generated
      )
    """).toPandas()

    if df_commercial.empty:
        print("No unprocessed Commercial rows found.")
    else:
        print(f"{len(df_commercial)} Commercial rows to process.")

    # ---------------------
    # add in commerical prompt
    # ---------------------

    commercial_prompt = """
    # Role
    You are a deinal appeal expert for a hospital. You are tasked with writing an inpatient lack of medical neccesity denial letter for commercial payors.

    # Task
    Complete a commercial medical necessity appeal letter template using the provided medical records. The patient’s claim was denied for an alleged lack of medically necissity.
    # Patient dischage summary text
    - Use the following text as the patient discharge summary:
    {discharge_summary}

    # HP Summary text
    - Use the following text as the patient HP summary: {hp_summary}

    # Treatment provided text
    - Use the following text as the treatment provided summary. Tell the story on why they needed to stay. Use lab results at the beginning of admission and at the end to fully tell why they needed to stay and be inpatient.
    {treatment_provided_text}

    # Patient Info
    - Name: {formatted_name}
    - DOB: {formatted_birthdate}
    - Payor ID: {policy_number}
    - Number of Midnights: {num_midnights}
    - Discharge Summary: {discharge_summary}
    - Claim # : {claim_number}
    - Tax ID: {tax_id}

    # MCG Documentation
    - Use the following MCG document as a reference for determining whether the patient met criteria for inpatient admission:
        MCG Document Name: {mcg_document_name}
        MCG Document Text: {mcg_document_text}

    **Key Requirements:**
    1. If any information is unavailable, set its value to null.
    2. For patient summary: draft concise, medically supported paragraphs explaining why, at the time of admission, the physician reasonably anticipated inpatient stay
    3. Emphasize specific clinical factors, patient acuity, and medical decision-making that justify inpatient admission
    4. Lab results should be presented in chronological order and highlight any abnormal or clinically significant trends that justify the level of care.
    - [hp_summary] is should have the physical examination when patient arrived in the ED and inital labs when patient was admitted
    5. Instead of saying "two-midnight stay" mention extended hospital stay
    6. At the end of the letter provide a nice short but descriptive summary of justification of the patients stay.
    7. Label the sections with Name: do not use any special characters outside a :


    Fill these keys:
    - add in {mapped_address} at the beginning of the letter format
    - patient_name: Exactly "{formatted_name}"
    - reason_for_admission: Extracted or summarized.
    - format "{total_charges}" in the actual dollar amount i.e if 15300.88 make it 15,300.88
    - facility name: exclude the word "Parent" in {facility_name}
    - Dates of Service: exactly {formatted_date_of_service}
    - Day Requested: {formatted_date_of_service_ip} Inpatient
    - Claim #: {claim_number}
    - if {day2_ip_date} is used please use format MM/DD/YYYY
    - For the [summary of care], write paragraphs to clearly document the clinical rationale supporting medical necessity, make sure you are thorough and specific.


    # Guidelines
    - Use precise medical and legal terminology suitable for commercial insurance claims reviewers
    - Prioritize clinical justification over administrative convenience
    - Maintain professional, objective tone throughout
    - Ensure the letter is concise, clear
    - Only use word "observation" only in the [treatment_provided_text], use monitoring instead if needed
    - Make sure to only include results that are related to the [formatted_date_of_service] dates
    - Make sure [summary_of_care] is in paragraph form and not bullet points

    # MCG Guidelines
    - You will be provided the relevant MCG guideline extracted using Azure Document Intelligence.
    - Read and summarize only the sections relevant to the patient's diagnosis or ICD code.
    - Pull MCG edition number from the [mcg_document_text]
    - Include short, medically precise sentences in the [summary_of_care] or [hospital_course] such as:
    "Patient met MCG 29th edition Clinical Indications for Admission to Inpatient Care under [mcg_document_name] due to [Clinical Indications for Admission to Inpatient Care]."
    (don't include .pdf in mcg_document_name)
    - Make sure to include all the patient has the required clinical findings to “check the box”
    - Use the [mcg_document_text] only to support or validate medical necessity

    # Response Format
    Return the final letter text, not json

    #Template
    {mapped_address}

    Reconsideration /Appeal for Denial of Inpatient Level of Care

    Patient Name: {formatted_name}
    DOB: {formatted_birthdate}
    Policy #: {policy_number}
    Claim #: {claim_number}
    Total Charges: $ {total_charges}
    Hospital Account #: {hsp_account_id}
    Dates of Service: {formatted_date_of_service}
    Day Requested: {formatted_date_of_service_ip} Inpatient
    Facility Name: {facility_name}
    NPI: {npi}
    Tax ID: {tax_id}

    Attention Reconsiderations/Appeals:

    Mercy Hospital is formally appealing the decision to deny Inpatient level of care for {formatted_name}, and instead approve only Observation services. Mercy is requesting Inpatient approval and reimbursement on {formatted_date_of_service}.

    [treatment_provided_text]

    The attending physician determined that inpatient care was medically necessary based on their professional assessment and the patient's medical history.

    [hospital_course] [hp_summary to help support the hospital course]

    [summary_of_care] + provide earliest supported lab results or vitals from [hp_summary] + [discharge_summary] and any lab results that changed over the course of the stay.

    [mcg guidelines here]

    Please provide a written outcome determination within 30 days and send your response to:

    Mercy Hospital
    Payor Audits and Denials Department
    Attention: Compliance Manager
    2115 South Fremont Ave.- Suite LL1
    Springfield, MO 65804

    Fax: 314-364-6231

    Respectfully,
    """

    def generate_and_save_letter_commercial(row):
        # Extract your fields
        mapped_address = row.get('mapped_address')
        formatted_name = row.get('formatted_name')
        formatted_birthdate = row.get('formatted_birthdate')
        hsp_account_id = row.get('hsp_account_id')
        policy_number = row.get('policy_number')
        num_midnights = row.get('number_of_midnights')
        discharge_summary = row.get('discharge_summary_text')
        hp_summary = row.get('hp_note_text')
        facility_name = row.get('facility_name')
        total_charges = row.get('total_charges')
        admission_date = row.get('admission_date')
        formatted_date_of_service = row.get('formatted_date_of_service')
        formatted_date_of_service_ip = row.get('formatted_date_of_service_ip')
        claim_number = row.get('claim_number')
        tax_id = row.get('tax_id')
        npi = row.get('npi')
        op_ip_flag = row.get('op_ip_flag')
        day2_ip_date = row.get('day2_ip_date')
        mcg_document_name = row.get('file_name')
        mcg_document_text = row.get('text')

        if op_ip_flag == "yes":
            treatment_provided_text = f"""
            Clinical Presentation: Upon emergent arrival at Mercy Hospital, {formatted_name} was originally placed in observation services for brief [reason_for_admission]. On {day2_ip_date}, {formatted_name} was admitted as an inpatient due to inability to be discharged after one midnight as {formatted_name} required continued hospital care.
            """
            logit_bias = None
        else:
            treatment_provided_text = f"""
            Clinical Presentation: Upon emergent arrival at Mercy Hospital, {formatted_name} was admitted as an inpatient for [reason_for_admission] + any supported lab results or vitals from [hp_summary]. The admitting Provider had a reasonable expectation of a 2 midnight stay due to
            """
            obs_tokens = enc.encode('observation')
            obs_tokens += enc.encode('Observation')

            logit_bias = {str(token_id): -100 for token_id in obs_tokens}

        formatted_prompt = commercial_prompt.format(
            mapped_address=mapped_address,
            formatted_name=formatted_name,
            formatted_birthdate=formatted_birthdate,
            hsp_account_id=hsp_account_id,
            policy_number=policy_number,
            num_midnights=num_midnights,
            discharge_summary=discharge_summary,
            hp_summary=hp_summary,
            facility_name=facility_name,
            total_charges=total_charges,
            admission_date=admission_date,
            formatted_date_of_service=formatted_date_of_service,
            formatted_date_of_service_ip=formatted_date_of_service_ip,
            claim_number=claim_number,
            tax_id=tax_id,
            npi=npi,
            day2_ip_date=day2_ip_date,
            treatment_provided_text=treatment_provided_text,
            mcg_document_name=mcg_document_name,
            mcg_document_text=mcg_document_text
        )

        messages_commercial = [
            {"role": "system", "content": formatted_prompt}
        ]

        # Make the API call
        response_object_commercial = client.chat.completions.create(
            messages=messages_commercial,
            model=model,
            temperature=0,
            max_tokens=5000,
            logit_bias=logit_bias
        )

        # Extract and return the response text
        return response_object_commercial.choices[0].message.content

    # Apply function and store result in new column
    # df_commercial = df_commercial.sample(15)
    df_commercial['letter_text'] = df_commercial.apply(generate_and_save_letter_commercial, axis=1)
    df_commercial['letter_curated_date'] = pd.to_datetime("today").normalize()
    df_commercial['letter_type'] = 'commercial'

    # -------------------------
    # merge the two dataframes
    # --------------------------
    if not df_medicare.empty or not df_commercial.empty:
        df_merged = pd.concat([df_medicare, df_commercial], ignore_index=True).sort_values(by="workqueue_entry_date").reset_index(drop=True)
    else:
        df_merged = pd.DataFrame(columns=["workqueue_entry_date", "hsp_account_id", "pat_mrn_id", "discharge_summary_note_id",
                                          "discharge_note_csn_id", "hp_note_id", "hp_note_csn_id", "letter_type", "letter_text", "letter_curated_date"])
        print("No unprocessed rows to merge.")

    # -----------------------------------------------------------------------
    # insert columns into table, schema was already created in featurization
    # -----------------------------------------------------------------------
    inference_score = df_merged[["workqueue_entry_date", "hsp_account_id", "pat_mrn_id", "discharge_summary_note_id","discharge_note_csn_id", "hp_note_id", "hp_note_csn_id", "letter_type", "letter_text", "letter_curated_date"]]
    if inference_score.empty:
        print("No rows to write to Spark table. Skipping.")
    else:
        spark_df = spark.createDataFrame(inference_score)
        spark_df = spark_df.withColumn("insert_tsp", current_timestamp())
        spark_df.write.mode("append").saveAsTable(f"{trgt_cat}.fin_ds.popsicle_inference_score")

    # ----------------------------------------------------------
    # transfer ownership to ds group in dev
    # -----------------------------------------------------------

    # transfer ownership to ds group in dev
    if trgt_cat == 'dev':
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()

        # Get all project tables **not** already owned by data-scientist group
        tbls_to_transfer = get_project_tbls(
            tbl_prefix = 'popsicle',
            schema = 'fin_ds',
            trgt_cat = trgt_cat,
            client = client,
            owners = ['res.az.idp-dev-001.datascientist'],
            exclude_owners = True
        )

        # Transfer ownership over to the data scientist group, hard-code dev bc this should only happen in dev
        list(map(
            lambda x: client.tables.update('dev.' + x, owner='res.az.idp-dev-001.datascientist'),
            tbls_to_transfer
        ))
else:
    print("no new rows added")