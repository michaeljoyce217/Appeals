#---------------------
# Imports &amp; settings
#---------------------
# %pip install openpyxl
# %pip install azure-ai-documentintelligence==1.0.2
# %pip install databricks-vectorsearch==0.57
# %restart_python

import os
import time
import re
from datetime import datetime
import pandas as pd

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.vector_search.client import VectorSearchClient

from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType, DateType
from utils.helper import get_project_tbls, transfer_ownership, extract_codes_from_end
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from itertools import groupby


# ---------------------
# Setup
# ---------------------
trgt_cat = os.environ.get('trgt_cat', 'dev')  # default to dev if not set
src_cat = 'prod'

if trgt_cat == 'dev':
    warehouse_id = 'e61c3334533dcb6b'
elif trgt_cat == 'test':
    warehouse_id = 'f41aeb4c5dd541f7'
elif trgt_cat == 'prod':
    warehouse_id = '9d00dc7a675a3706'
else:
    raise ValueError(f"Unknown target catalog: {trgt_cat}")

# create popiscle inference table
tbl_exists = spark.catalog.tableExists(f'{trgt_cat}.fin_ds.popsicle_inference')

popsicle_featurization_schema = StructType([StructField("pat_id", StringType(), False),  # not null
                                    StructField("pat_mrn_id", StringType(), False),  # not null
                                    StructField("hsp_account_id", StringType(), False),  # not null
                                    StructField("workqueue_entry_date", StringType(), False),  # not null
                                    StructField("letter_type", StringType(), False),  # not null
                                    StructField("formatted_name", StringType(), False),  # not null
                                    StructField("formatted_birthdate", StringType(), False),  # not null
                                    StructField("admission_date", StringType(), False),  # not null
                                    StructField("inp_admission_date", StringType(), False),  # not null
                                    StructField("inp_discharge_date", StringType(), False),  # not null
                                    StructField("age_at_discharge", DoubleType(), False),  # not null
                                    StructField("formatted_date_of_service_ip", StringType(), True),  # not null
                                    StructField("formatted_date_of_service", StringType(), True),  # not null
                                    StructField("number_of_midnights", IntegerType(), False),  # not null
                                    StructField("coverage_id", StringType(), True),  # nullable
                                    StructField("payor_id", StringType(), True),  # nullable
                                    StructField("plan_id", StringType(), True),  # nullable
                                    StructField("policy_number", StringType(), True),  # nullable
                                    StructField("discharge_summary_note_id", StringType(), True),  # nullable
                                    StructField("discharge_note_csn_id", StringType(), True),
                                    StructField("discharge_summary_text", StringType(), True),  # nullable
                                    StructField("hp_note_id", StringType(), True),  # nullable
                                    StructField("hp_note_csn_id", StringType(), True),  # nullable
                                    StructField("hp_note_text", StringType(), True),  # nullable
                                    StructField("facility_name", StringType(), True),  # nullable
                                    StructField("total_charges", DoubleType(), True),  # nullable
                                    StructField("tax_id", StringType(), True),  # nullable
                                    StructField("npi", StringType(), True),  # nullable
                                    StructField("claim_number", StringType(), True),  # nullable
                                    StructField("insert_tsp", TimestampType(), False),
                                    StructField("day1_arrival_date", DateType(), True),
                                    StructField("day2_ip_date", DateType(), True),
                                    StructField("op_ip_flag", StringType(), False),
                                    StructField('mapped_address', StringType(), True),
                                    StructField('code', StringType(), True),
                                    StructField('dx_name', StringType(), True),
                                    StructField('icd10_proc_code', StringType(), True),
                                    StructField('px_name', StringType(), True),
                                    StructField('file_name', StringType(), True),
                                    StructField('mcg_text', StringType(), True)
])

df = pd.DataFrame(columns=[field.name for field in popsicle_featurization_schema.fields])
spark.createDataFrame([], popsicle_featurization_schema) \
    .write.mode('append') \
    .option("mergeSchema", "true") \
    .saveAsTable(f'{trgt_cat}.fin_ds.popsicle_inference')

# create inference score table
is_tbl_exists = spark.catalog.tableExists(f'{trgt_cat}.fin_ds.popsicle_inference_score')

if not is_tbl_exists:
    popsicle_inference_score_schema = StructType([
        StructField("workqueue_entry_date", StringType(), False),  # not null
        StructField("hsp_account_id", StringType(), False),  # not null
        StructField("pat_mrn_id", StringType(), False),  # not null
        StructField("discharge_summary_note_id", StringType(), True),  # nullable
        StructField("discharge_note_csn_id", StringType(), True),
        StructField("hp_note_id", StringType(), True),  # nullable
        StructField("hp_note_csn_id", StringType(), True),
        StructField("letter_type", StringType(), False), # nullable
        StructField("letter_text", StringType(), False), # not null
        StructField("letter_curated_date", TimestampType(), False), # not null
        StructField("insert_tsp", TimestampType(), False)
    ])
    df_score = pd.DataFrame(columns=[field.name for field in popsicle_inference_score_schema.fields])
    spark.createDataFrame([], popsicle_inference_score_schema).write.mode('append').saveAsTable(f'{trgt_cat}.fin_ds.popsicle_inference_score')

# -----------------------------
# Add in Addresses for Payors
# -----------------------------

# load file and make df
address_df = pd.read_excel("../utils/mapped_payer_appeal_address 10-24-25.xlsx", engine='openpyxl')
spark_address_df = spark.createDataFrame(address_df.astype(str)).write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{trgt_cat}.fin_ds.popsicle_address_map_new")

#------------------------------
# add in mcg documents table
#-------------------------------
# check if table exsits
mcg_tbl = f"{trgt_cat}.fin_ds.popsicle_mcg_documents"
mcg_tbl_exists = spark.catalog.tableExists(f'{trgt_cat}.fin_ds.popsicle_mcg_documents')

# set `` and `` variables with the values from the Azure portal
key = "DUMMY_AZURE_DOC_INTEL_KEY"
endpoint= "DUMMY_AZURE_DOC_INTEL_ENDPOINT"
# key = dbutils.secrets.get(scope = 'idp_etl', key = "az-aidcmntintel-key1") # what endpoint is this to?
path_to_doc = "../utils/mcg_documents"

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# if table excists, load file to avoid dups
if mcg_tbl_exists:
    existing_df = spark.table(f"{trgt_cat}.fin_ds.popsicle_mcg_documents")
    existing_files = set(row.file_name for row in existing_df.select("file_name").collect())
else:
    existing_files = set()
# look to see new documnts
records = []

for file_name in os.listdir(path_to_doc):
    # Skip non-document files
    if not file_name.lower().endswith((".pdf", ".docx", ".jpg", ".png")):
        continue

    # Skip already-processed files
    if file_name in existing_files:
        continue

    file_path = os.path.join(path_to_doc, file_name)
    print(f"Processing new document: {file_name}")

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", body=f)
    result = poller.result()

    full_text = result.content

    codes_list = extract_codes_from_end(full_text)

    records.append({
        "file_name": file_name,
        "text": full_text,
        "codes": ",".join(codes_list) if codes_list else ""
    })

if not mcg_tbl_exists:
    # Create table FIRST time
    print("Creating brand-new popsicle_mcg_documents table...")

    mcg_schema = StructType([
        StructField("file_name", StringType(), True),
        StructField("text", StringType(), True),
        StructField("codes", StringType(), True)
    ])

    if not records:
        empty_df = spark.createDataFrame([], schema=mcg_schema)
        empty_df.write.mode("overwrite").saveAsTable(f"{trgt_cat}.fin_ds.popsicle_mcg_documents")
        print("Created empty table (no documents found).")

    else:
        df_docs = spark.createDataFrame(pd.DataFrame(records))
        df_docs.write.mode("overwrite").saveAsTable(f"{trgt_cat}.fin_ds.popsicle_mcg_documents")
        print(f"Table created with {df_docs.count()} documents.")

else:
    # APPEND ONLY NEW DOCUMENTS
    if records:
        df_new = spark.createDataFrame(pd.DataFrame(records))
        df_new.write.mode("append").saveAsTable(f"{trgt_cat}.fin_ds.popsicle_mcg_documents")
        print(f"Added {df_new.count()} NEW documents.")
    else:
        print("No new documents to add. Table already up to date.")

# ---------------------
# Core query
# ---------------------
default_date = "2025-07-01"

base_select = f"""
with latest_date as (
select coalesce(max(workqueue_entry_date), date '{default_date}') as last_date
from {trgt_cat}.fin_ds.popsicle_inference_score
)

,medicare_pop as (
SELECT HQ.HSP_ACCOUNT_ID
,cast(to_timestamp(enter_queue_dttm) as date) as WORKQUEUE_ENTRY_DATE
,workqueue_id
,epm.financial_class
FROM
  clarity.hsp_Acct_workqueue_list hq
  LEFT JOIN clarity.hsp_account ha on hq.hsp_account_id = ha.hsp_account_id
  LEFT JOIN clarity.CLARITY_EPM EPM ON HA.PRIMARY_PAYOR_ID = EPM.PAYOR_ID
  cross join latest_date l
WHERE
--Account WQ Mercy HB Corro Health Appeals
 hq.WORKQUEUE_ID in (1315223)
 --this filters to accounts that hit the workqueue 5 days ago but modify as you see fit
and (
    cast(to_timestamp(enter_queue_dttm) as date) > l.last_date
    OR hq.hsp_account_id NOT IN (
        SELECT hsp_account_id
        FROM {trgt_cat}.fin_ds.popsicle_inference
    )
)
and cast(to_timestamp(enter_queue_dttm) as date) <= current_date - 5
 --PAYORS FOR PILOT - United Healthcare Medicare Advantage Contracted and Humana Medicare Advantage Contracted
 and epm.FINANCIAL_CLASS in (115, 116)
)

,commerical_pop as (
SELECT HQ.HSP_ACCOUNT_ID
,cast(to_timestamp(enter_queue_dttm) as date) as WORKQUEUE_ENTRY_DATE
,workqueue_id
,epm.financial_class
FROM
  clarity.hsp_Acct_workqueue_list hq
  LEFT JOIN clarity.hsp_account ha on hq.hsp_account_id = ha.hsp_account_id
  LEFT JOIN clarity.CLARITY_EPM EPM ON HA.PRIMARY_PAYOR_ID = EPM.PAYOR_ID
  cross join latest_date l
WHERE
--Account WQ Mercy HB Corro Health Appeals
 hq.WORKQUEUE_ID in (1315223)
--this filters to accounts that hit the workqueue 5 days ago but modify as you see fit
and (
       cast(to_timestamp(enter_queue_dttm) as date) > l.last_date
       OR hq.hsp_account_id NOT IN (
            SELECT hsp_account_id
            FROM {trgt_cat}.fin_ds.popsicle_inference
       )
   )
and cast(to_timestamp(enter_queue_dttm) as date) <= current_date - 5
--PAYORS FOR PILOT - commercial contracts
 and epm.FINANCIAL_CLASS in (100,1,109,110,111,112,106,113, 114) --adding new fc
)

,pop as (
select * from medicare_pop
union all
select * from commerical_pop
)

,notes as (
select * from (
select
peh.pat_id,
peh.hsp_account_id,
nte.ip_note_type,
nte.note_id,
nte.note_csn_id,
nte.contact_date as note_contact_date,
nte.ent_inst_local_dttm as entry_datetime,
concat_ws("\n", sort_array(collect_list(struct(line, note_text))).note_text) as note_text
from clarity_cur.pat_enc_hsp_har_enh peh -- only encounters that are for hospital accounts (no outpatient)
inner join clarity_cur.hno_note_text_enh nte using(pat_enc_csn_id)
where
ip_note_type in ('Discharge Summary', 'H&amp;P')
--and note_status = 'Signed'
group by peh.pat_id, peh.hsp_account_id, nte.ip_note_type, nte.note_id, nte.note_csn_id, nte.contact_date, nte.ent_inst_local_dttm
)
qualify row_number() over (partition by hsp_account_id, ip_note_type order by note_contact_date desc, entry_datetime desc) = 1
)

,hp_note as (
select
pat_id
,hsp_account_id
,note_csn_id as hp_note_csn_id
,note_id as hp_note_id
,note_text as HP_note_text
from notes
where ip_note_type = 'H&amp;P'
)

,discharge as (
select
pat_id
,hsp_account_id
,note_csn_id as discharge_note_csn_id
,note_id as discharge_summary_note_id
,note_text as discharge_summary_text
from notes
where ip_note_type = 'Discharge Summary'
)

,claim_taxid as (
select
hsp_account_id,
max(TAX_ID) as TAX_ID,
max(BIL_PROV_NPI) as BIL_PROV_NPI,
string_agg(ICN_NO, ', ') as claim_number,
string_agg(ICN, ', ') as icn_value,
string_agg(INT_CONTROL_NUMBER, ', ') as int_control_number,
array_join(array_distinct(filter(array(any_value(ICN),any_value(ICN_NO),any_value(INT_CONTROL_NUMBER)), x -> x is not null)),', ') as all_claim_ids
from (select HOSPITAL_ACCT_ID as hsp_account_id, TAX_ID, BIL_PROV_NPI, ICN_NO, ICN, INT_CONTROL_NUMBER
from clarity.HSP_CLAIM_DETAIL2
left join clarity.bdc_info using(CLAIM_PRINT_ID)
left join clarity.hsp_claim_detail1 using(CLAIM_PRINT_ID)
left join clarity.clm_values on clm_values.RECORD_ID = HSP_CLAIM_DETAIL2.CLM_EXT_VAL_ID
left join fin_rev_cycle_sem.rev_assur_claim_dnl on rev_assur_claim_dnl.CLAIM_RCRD_ID = clm_values.RECORD_ID
left join clarity.cl_rmt_clm_info on cl_rmt_clm_info.INV_NO = rev_assur_claim_dnl.INV_NBR
where bdc_info.BDC_ID is not null and bdc_info.RECORD_STATUS_C not in (90,99)
) sub
group by hsp_account_id
)

,proc_code as (
  select px.hsp_account_id
  ,px.line
  ,px.ref_bill_code as icd10_proc_code
  ,px.procedure_name as px_name
  ,REF_BILL_CODE_SET

  from prod.clarity_cur.hsp_acct_px_list_enh px
  inner join pop using(hsp_account_id)
  where REF_BILL_CODE_SET = 'ICD-10-PCS'  and line = 1
)

,final as (select
patient.pat_id
,patient.pat_mrn_id
,ha.hsp_account_id
,WORKQUEUE_ENTRY_DATE as workqueue_entry_date
,case when (workqueue_id = 1315223) and (p.financial_class in (115, 116)) then 'Medicare'
when (workqueue_id = 1315223) and (p.financial_class in (100,1,109,110,111,112,106,113,114)) then 'Commercial'
else null end as letter_type
,concat(patient.pat_first_name, ' ',patient.pat_last_name) as formatted_name
,date_format(patient.BIRTH_DATE, 'MM/dd/yyyy') as formatted_birthdate
,date_format(ha.adm_date_time, 'MM/dd/yyyy') admission_date
,date_format(ha.ip_admit_date_time, 'MM/dd/yyyy') inp_admission_date
,date_format(ha.disch_date_time, 'MM/dd/yyyy') inp_discharge_date
,floor(datediff(ha.disch_date_time, patient.birth_date)/365.25) as age_at_discharge
,concat(date_format(ha.ip_admit_date_time, 'MM/dd/yyyy'), '-', date_format(ha.disch_date_time, 'MM/dd/yyyy')) as formatted_date_of_service_ip
,concat(date_format(ha.adm_date_time, 'MM/dd/yyyy'), '-', date_format(ha.disch_date_time, 'MM/dd/yyyy')) as formatted_date_of_service
,datediff(ha.disch_date_time, date(ha.adm_date_time)) number_of_midnights
,ha.coverage_id
,cov.payor_id
,cov.plan_id
,cov.subscr_num as policy_number
,case when discharge_summary_note_id is null then 'no id available' else discharge_summary_note_id end discharge_summary_note_id
,case when discharge_note_csn_id is null then 'no id available' else discharge_note_csn_id end discharge_note_csn_id
,case when discharge_summary_text is null then 'No Note Available' else discharge_summary_text end discharge_summary_text
,case when hp_note_id is null then 'no id available' else hp_note_id end hp_note_id
,case when hp_note_csn_id is null then 'no id available' else hp_note_csn_id end hp_note_csn_id
,case when HP_note_text is null then 'No Note Available' else HP_note_text end HP_note_text
,initcap(trim(regexp_replace(ha.loc_name, '^PARENT\\s+', ''))) as facility_name
,tot_chgs as total_charges
,t.TAX_ID as tax_id
,t.BIL_PROV_NPI npi
,t.all_claim_ids as claim_number
,am.payor_addresses as mapped_address
,proc_code.icd10_proc_code
,proc_code.px_name

from pop p
inner join clarity_cur.hsp_account_enh ha on p.hsp_account_id = ha.hsp_account_id
inner join clarity_cur.patient_enh patient on ha.pat_id = patient.pat_id
left join discharge d on p.hsp_account_id = d.hsp_account_id
left join hp_note h on p.hsp_account_id = h.hsp_account_id
inner join clarity.coverage cov on ha.coverage_id = cov.coverage_id
left join proc_code on p.hsp_account_id = proc_code.hsp_account_id
inner join claim_taxid t on p.hsp_account_id = t.hsp_account_id
left join {trgt_cat}.fin_ds.popsicle_address_map_new am
  on ((am.map_to = 'Payor' and (cov.payor_id = am.mapped_payor_id or cov.plan_id = am.mapped_plan_id)
  or (am.map_to = 'Plans' and cov.plan_id = am.mapped_plan_id or cov.payor_id = am.mapped_payor_id)
  or (am.map_to = 'Payor or plan' and (cov.plan_id = am.mapped_plan_id or cov.payor_id = am.mapped_payor_id))
  or (am.map_to = 'Payor and Location?' and cast(t.BIL_PROV_NPI as string) in (
            'DUMMY_NPI_01','DUMMY_NPI_02','DUMMY_NPI_03','DUMMY_NPI_04','DUMMY_NPI_05',
            'DUMMY_NPI_06','DUMMY_NPI_07','DUMMY_NPI_08','DUMMY_NPI_09') and cov.payor_id = am.mapped_payor_id) -- AR

  or (am.map_to = 'Payor and Location?' and cast(t.BIL_PROV_NPI as string) in (
            'DUMMY_NPI_10','DUMMY_NPI_11','DUMMY_NPI_12','DUMMY_NPI_13','DUMMY_NPI_14',
            'DUMMY_NPI_15','DUMMY_NPI_16','DUMMY_NPI_17','DUMMY_NPI_18','DUMMY_NPI_19',
            'DUMMY_NPI_20','DUMMY_NPI_21','DUMMY_NPI_22','DUMMY_NPI_23','DUMMY_NPI_24',
            'DUMMY_NPI_25','DUMMY_NPI_26','DUMMY_NPI_27', 'DUMMY_NPI_28', 'DUMMY_NPI_29'
            ,'DUMMY_NPI_30') and cov.payor_id = am.mapped_payor_id) -- MO

  or (am.map_to = 'Payor and Location?' and cast(t.BIL_PROV_NPI as string) in (
            'DUMMY_NPI_31','DUMMY_NPI_32','DUMMY_NPI_33','DUMMY_NPI_34',
            'DUMMY_NPI_35','DUMMY_NPI_36','DUMMY_NPI_37','DUMMY_NPI_38',
            'DUMMY_NPI_39','DUMMY_NPI_40','DUMMY_NPI_41','DUMMY_NPI_42',
            'DUMMY_NPI_43','DUMMY_NPI_44','DUMMY_NPI_45') and cov.payor_id = am.mapped_payor_id) -- OK
      )
) qualify row_number() over (partition by ha.hsp_account_id, t.BIL_PROV_NPI order by ha.hsp_account_id)= 1
)

,adt as (
select
  ca.pat_id,
  ca.pat_enc_csn_id,
  ca.event_time,
  cast(ca.event_time as date) as event_dt,
  ca.order_id,
  proc.proc_id,
  eap.proc_name,
  ca.pat_class_c,
  hsp.hsp_account_id,
  hsp.ADT_PAT_CLASS_C
FROM clarity.clarity_adt ca
left join clarity.order_proc proc
  on proc.ORDER_PROC_ID = ca.ORDER_ID
left join clarity.clarity_eap eap
  on eap.proc_id = proc.proc_id
left join clarity.pat_enc_hsp hsp
on ca.pat_enc_csn_id = hsp.pat_enc_csn_id
inner join final using(pat_id)
  where ca.pat_enc_csn_csn_id is not null
  and ca.event_time between (select min(workqueue_entry_date) from final) and (select max(workqueue_entry_date) from final)
)

--find the arrival date/time for each encounter. earliest event per encounter = arrival
,arrival as (
  select pat_id,
  pat_enc_csn_id,
  hsp_account_id,
  min(event_time) as arrival_TS,
  cast(min(event_time) as date) as arrival_dt
  from adt
  group by pat_id, pat_enc_csn_id, hsp_account_id
)
--identify encounters where day 0 (arrival day) had an observation/outpatient order
--proc id in 10587180,10587181,10587182 - these align with the adt proc CODES
,day1_op_obs as (
  select DISTINCT
    a.pat_id,
    a.pat_enc_csn_id,
    a.hsp_account_id,
    d.ADT_PAT_CLASS_C
  from adt d
  left join arrival a
   on a.pat_id = d.pat_id
   and a.pat_enc_csn_id = d.pat_enc_csn_id
   where d.event_dt = a.arrival_dt
   and d.proc_id in (10587180,10587181,10587182)
)
--identify encounters where day 1 (day after arrival) had inpatient status (pat class 101)
,day2_ip as (
  select DISTINCT
    a.pat_id,
    a.pat_enc_csn_id,
    a.hsp_account_id,
    d.ADT_PAT_CLASS_C
  FROM adt d
  left join arrival a
    on a.pat_id = d.pat_id
    and a.pat_enc_csn_id = d.pat_enc_csn_id
  where d.event_dt = dateadd(day, 1, a.arrival_dt)
  and (d.proc_id in (10303742,263)
  or d.pat_class_c = '101')
)

--may need to identify encounters where day after arrival there is an inpatient order placed.

--final output - encounters that start with ip/obs order on day 0 and transition to ip status on day 1
,op_ipencounters as (SELECT DISTINCT
ar.pat_enc_csn_id,
ar.hsp_account_id,
ar.arrival_dt as day1_arrival_date,
cast(dateadd(day, 1, ar.arrival_dt) as  date) as day2_ip_date
from arrival as ar
LEFT JOIN day1_op_obs op
  ON ar.pat_id = op.pat_id
  and ar.pat_enc_csn_id = op.pat_enc_csn_id
LEFT JOIN day2_ip ip
  ON ar.pat_id = ip.pat_id
  and ar.pat_enc_csn_id = ip.pat_enc_csn_id
  where --ar.pat_id in ('Z13088195','Z695811','Z12246400') and
   op.ADT_PAT_CLASS_C = '101'
)

,mcg_exploded as (
  select
    mcg.*,
    trim(code) as mcg_code
    ,array_except(filter(split(regexp_replace(lower(file_name), '[^a-z0-9 ]', ''),' +'),
        w -> w NOT IN ('the','a','an','and','or','with','without','unspecified',
        'other','cms','hcc','of','in','on','disease','illness') AND length(w) > 2),
        array()) AS file_tokens
  from {trgt_cat}.fin_ds.popsicle_mcg_documents mcg
  lateral view explode(split(mcg.codes, ',')) t as code
)

,cleaned as (select
final.pat_id
,final.pat_mrn_id
,final.hsp_account_id
,final.workqueue_entry_date
,final.letter_type
,final.formatted_name
,final.formatted_birthdate
,final.admission_date
,final.inp_admission_date
,final.inp_discharge_date
,final.age_at_discharge
,final.formatted_date_of_service_ip
,final.formatted_date_of_service
,final.number_of_midnights
,final.coverage_id
,final.payor_id
,final.plan_id
,final.policy_number
,final.discharge_summary_note_id
,final.discharge_note_csn_id
,final.discharge_summary_text
,final.hp_note_id
,final.hp_note_csn_id
,final.hp_note_text
,final.facility_name
,final.total_charges
,final.tax_id
,final.npi
,final.claim_number
,current_date() as insert_tsp
,day1_arrival_date
,day2_ip_date
,dx.code
,dx.dx_name
,mcg_exploded.file_name
,mcg_exploded.mcg_code
,mcg_exploded.text as mcg_text
,array_except(filter(split(regexp_replace(lower(dx.dx_name), '[^a-z0-9 ]', ''), ' +'),
        w -> w NOT IN ('the','a','an','and','or','with','without','unspecified',
        'other','cms','hcc','of','in','on','disease','illness') AND length(w) > 2),
        array()) AS dx_tokens
-- ,array_except(filter(split(regexp_replace(lower(mcg_exploded.file_name), '[^a-z0-9 ]', ''),' +'),
--         w -> w NOT IN ('the','a','an','and','or','with','without','unspecified',
--         'other','cms','hcc','of','in','on','disease','illness') AND length(w) > 2),
--         array()) AS file_tokens
,file_tokens
,final.icd10_proc_code
,final.px_name
,cast((case when op_ipencounters.pat_enc_csn_id is not null then 'yes' else 'no' end) as string) as op_ip_flag
,final.mapped_address
from final
left outer join op_ipencounters using(hsp_account_id)
left join prod.clarity_cur.hsp_acct_dx_list_enh dx using(hsp_account_id)
left join mcg_exploded on
  (mcg_exploded.mcg_code = dx.code OR
  mcg_exploded.mcg_code = final.icd10_proc_code OR
  size(array_intersect(array_except(filter(split(regexp_replace(lower(dx.dx_name), '[^a-z0-9 ]', ''), ' +'),
                w -> w NOT IN (
                  'the','a','an','and','or','with','without','unspecified','other',
                  'cms','hcc','of','in','on','disease','illness'
                ) AND length(w) > 2),
              array()),file_tokens)) > 0)
where dx.line = 1
-- order by workqueue_entry_date
)

,scored as (
select
*
-- number of shared tokens
,size(array_intersect(dx_tokens, file_tokens)) AS shared_tokens
-- similarity score = shared tokens / dx token count
,case when size(dx_tokens) > 0 then size(array_intersect(dx_tokens, file_tokens)) / size(dx_tokens)
  else 0 end as similarity_score
,case when mcg_code = cleaned.code then 1.0
  when mcg_code = icd10_proc_code then 0.8
  else 0 end as code_match_score
, (case when mcg_code = cleaned.code then 1.0       -- strongest
  when mcg_code = icd10_proc_code then 0.8
  when mcg_code = px_name then 0.6
  else 0.2 * (case when size(dx_tokens) > 0 then size(array_intersect(dx_tokens, file_tokens))/ size(dx_tokens) else 0 end
  ) end) AS combined_score
from cleaned
)

,ranked as (
select
*
,row_number() over (partition by hsp_account_id order by combined_score DESC, similarity_score DESC, size(file_tokens) DESC) as rn
from scored
)

select
pat_id
,pat_mrn_id
,hsp_account_id
,workqueue_entry_date
,letter_type
,formatted_name
,formatted_birthdate
,admission_date
,inp_admission_date
,inp_discharge_date
,age_at_discharge
,formatted_date_of_service_ip
,formatted_date_of_service
,number_of_midnights
,coverage_id
,payor_id
,plan_id
,policy_number
,discharge_summary_note_id
,discharge_note_csn_id
,discharge_summary_text
,hp_note_id
,hp_note_csn_id
,hp_note_text
,facility_name
,total_charges
,tax_id
,npi
,claim_number
,insert_tsp
,day1_arrival_date
,day2_ip_date
,code
,dx_name
,file_name
,mcg_text
,icd10_proc_code
,px_name
,op_ip_flag
,mapped_address
from ranked
where rn = 1
order by workqueue_entry_date

"""

## if this fails and make a pandas dataframe and do what I am doing in line 287 and then do what same process but with spark and not pure sql
# get set of rows in new table that don't yet exist in existing table (in a pandas df), convert to spark table, do append operation with merge schema

# ---------------------
# Create table if not exists
# ---------------------
create_sql = f"""
CREATE TABLE IF NOT EXISTS {trgt_cat}.fin_ds.popsicle_inference AS
{base_select}
"""

# ---------------------
# Append new rows
# ---------------------
append_sql = f"""
INSERT INTO {trgt_cat}.fin_ds.popsicle_inference (
    pat_id,
    pat_mrn_id,
    hsp_account_id,
    workqueue_entry_date,
    letter_type,
    formatted_name,
    formatted_birthdate,
    admission_date,
    inp_admission_date,
    inp_discharge_date,
    age_at_discharge,
    formatted_date_of_service_ip,
    formatted_date_of_service,
    number_of_midnights,
    coverage_id,
    payor_id,
    plan_id,
    policy_number,
    discharge_summary_note_id,
    discharge_note_csn_id,
    discharge_summary_text,
    hp_note_id,
    hp_note_csn_id,
    hp_note_text,
    facility_name,
    total_charges,
    tax_id,
    npi,
    claim_number,
    insert_tsp,
    day1_arrival_date,
    day2_ip_date,
    op_ip_flag,
    mapped_address,
    code,
    dx_name,
    icd10_proc_code,
    px_name,
    file_name,
    mcg_text
)
SELECT
    src.pat_id,
    src.pat_mrn_id,
    src.hsp_account_id,
    src.workqueue_entry_date,
    src.letter_type,
    src.formatted_name,
    src.formatted_birthdate,
    src.admission_date,
    src.inp_admission_date,
    src.inp_discharge_date,
    src.age_at_discharge,
    src.formatted_date_of_service_ip,
    src.formatted_date_of_service,
    src.number_of_midnights,
    src.coverage_id,
    src.payor_id,
    src.plan_id,
    src.policy_number,
    src.discharge_summary_note_id,
    src.discharge_note_csn_id,
    src.discharge_summary_text,
    src.hp_note_id,
    src.hp_note_csn_id,
    src.hp_note_text,
    src.facility_name,
    src.total_charges,
    src.tax_id,
    src.npi,
    src.claim_number,
    current_timestamp(),
    src.day1_arrival_date,
    src.day2_ip_date,
    src.op_ip_flag,
    src.mapped_address,
    src.code,
    src.dx_name,
    src.icd10_proc_code,
    src.px_name,
    src.file_name,
    src.mcg_text
FROM ({base_select}) src
WHERE NOT EXISTS (
    SELECT 1
    FROM {trgt_cat}.fin_ds.popsicle_inference tgt
    WHERE tgt.hsp_account_id = src.hsp_account_id
      AND tgt.workqueue_entry_date = src.workqueue_entry_date
)
"""

# ---------------------
# Run queries in sequence
# ---------------------
w = WorkspaceClient()

def run_sql(statement):
    q_resp = w.statement_execution.execute_statement(statement=statement,warehouse_id=warehouse_id,catalog=src_cat,wait_timeout='0s')
    q_id = q_resp.statement_id
    current_status = q_resp.status.state.value
    time_polling = 0

    while current_status != 'SUCCEEDED' and time_polling < 300:
        time.sleep(5)
        status_resp = w.statement_execution.get_statement(q_id)
        current_status = status_resp.status.state.value
        if current_status == 'FAILED':
            raise Exception(f"query failed: {status_resp.status.error.message}")
        print(f'time polling: {time_polling}s | Status: {current_status}')
        time_polling += 5
    print("query completed")

# Create table first (only if doesn't exist yet)
print("Creating table if not exists...")
run_sql(create_sql)

# Always append new data
print("Appending new rows...")
run_sql(append_sql)


#transfer ownership to ds group in dev
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

