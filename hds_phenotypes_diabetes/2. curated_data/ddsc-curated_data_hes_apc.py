# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Data - HES APC
# MAGIC
# MAGIC **Description** This notebook curates a prepared version of HES APC for use in all subsequent notebooks.
# MAGIC
# MAGIC The following data cleaning has been done:
# MAGIC - HES APC filtered to the archived_on batch defined in paramters
# MAGIC - Wrangled into long format (codes per row)
# MAGIC - Any records dated 1800-01-01 or 1801-01-01 removed
# MAGIC - Records > than the last observable date removed
# MAGIC - Individual censor dates applied and for those who have died, records > date of death removed
# MAGIC - Null dates and PERSON_ID removed
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_data_hes_apc_{algorithm_timestamp}`**

# COMMAND ----------

# MAGIC %md
# MAGIC #Setup

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

import databricks.koalas as ks
import pandas as pd
import numpy as np

import re
import io
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-last_observable_date"

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

hes_apc = extract_batch_from_archive(parameters_df_datasets, 'hes_apc')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Lookups

# COMMAND ----------

icd_lookup = spark.table(f'dss_corporate.icd10_group_chapter_v01')

# COMMAND ----------

icd_lookup_clean = (
    icd_lookup
    .select(f.col("ALT_CODE").alias("CODE"), "ICD10_DESCRIPTION")
    .withColumn('CODE', f.regexp_replace('CODE', r'X$', ''))
    .withColumn('CODE', f.regexp_replace('CODE', r'[.,\-\s]', ''))
)

# COMMAND ----------

# MAGIC %md
# MAGIC #Curate

# COMMAND ----------

if individual_censor_dates_flag==True:
    individual_censor_dates = spark.table(individual_censor_dates_table)
else:
    individual_censor_dates = demographics.select("PERSON_ID",f.col("DATE_OF_DEATH").alias("CENSOR_END")).filter(f.col("CENSOR_END").isNotNull())

# COMMAND ----------

# Prepare long format of HES APC
_hes_apc = (
  hes_apc  
  .select(['PERSON_ID_DEID', 'EPIKEY', 'EPISTART', 'ADMIDATE'] 
          + [col for col in list(hes_apc.columns) if re.match(r'^DIAG_(3|4)_\d\d$', col)])
  .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')
  .orderBy('PERSON_ID', 'EPIKEY')
)

hes_apc_long = (
  reshape_wide_to_long_multi(_hes_apc, i=['PERSON_ID', 'EPIKEY', 'EPISTART', 'ADMIDATE'], j='POSITION', stubnames=['DIAG_4_', 'DIAG_3_'])
  .withColumn('_tmp', f.substring(f.col('DIAG_4_'), 1, 3))
  .withColumn('_chk', udf_null_safe_equality('DIAG_3_', '_tmp').cast(t.IntegerType()))
  .withColumn('_DIAG_4_len', f.length(f.col('DIAG_4_')))
  .withColumn('_chk2', f.when((f.col('_DIAG_4_len').isNull()) | (f.col('_DIAG_4_len') <= 4), 1).otherwise(0))
)

hes_apc_long = (
  hes_apc_long
  .drop('_tmp', '_chk')
)

hes_apc_long = reshape_wide_to_long_multi(hes_apc_long, i=['PERSON_ID', 'EPIKEY', 'EPISTART', 'ADMIDATE', 'POSITION'], j='DIAG_DIGITS', stubnames=['DIAG_'])\
  .withColumnRenamed('POSITION', 'DIAG_POSITION')\
  .withColumn('DIAG_POSITION', f.regexp_replace('DIAG_POSITION', r'^[0]', ''))\
  .withColumn('DIAG_DIGITS', f.regexp_replace('DIAG_DIGITS', r'[_]', ''))\
  .withColumn('DIAG_', f.regexp_replace('DIAG_', r'X$', ''))\
  .withColumn('DIAG_', f.regexp_replace('DIAG_', r'[.,\-\s]', ''))\
  .withColumnRenamed('DIAG_', 'CODE')\
  .where((f.col('CODE').isNotNull()) & (f.col('CODE') != ''))\
  .orderBy(['PERSON_ID', 'EPIKEY', 'DIAG_DIGITS', 'DIAG_POSITION'])


hes_apc_long = (
  hes_apc_long
  # .join(icd_lookup_clean,on="CODE",how="left") #code descriptions
  .withColumn("EPISTART", f.when((f.col("EPISTART") == "1800-01-01") | (f.col("EPISTART") == "1801-01-01"), None).otherwise(f.col("EPISTART")))
  .where(f.col('EPISTART').isNotNull())
  .filter(f.col("EPISTART")<last_observable_date)
  #filter out records > DATE_OF_DEATH
  .join(individual_censor_dates,on="PERSON_ID",how="left")
  .withColumn("CENSOR_END", f.when(f.col("CENSOR_END").isNull(),last_observable_date).otherwise(f.col("CENSOR_END")))
  .filter(f.col("EPISTART")<=f.col("CENSOR_END"))
  .drop("CENSOR_END")
)

# COMMAND ----------

if individual_censor_dates_flag==True:
    hes_apc_long = hes_apc_long
else:
    hes_apc_long = hes_apc_long.filter(f.col("EPISTART")<study_end_date)

# COMMAND ----------

# MAGIC %md
# MAGIC #Save

# COMMAND ----------

save_table(df=hes_apc_long, out_name=f'{proj}_curated_data_hes_apc_{algorithm_timestamp}', save_previous=False)