# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Data - GDPPR
# MAGIC
# MAGIC **Description** This notebook curates a prepared version of GDPPR for use in all subsequent notebooks.
# MAGIC
# MAGIC The following data cleaning has been done:
# MAGIC - GDPPR filtered to the archived_on batch defined in paramters
# MAGIC - Any records dated 1800-01-01 or 1801-01-01 removed
# MAGIC - Records > than the last observable date removed
# MAGIC - Individual censor dates applied and for those who have died, records > date of death removed
# MAGIC - Null dates and PERSON_ID removed
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_data_gdppr_{algorithm_timestamp}`**

# COMMAND ----------

# MAGIC %md #Setup

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

# MAGIC %md #Parameters

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-last_observable_date"

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

gdppr   = extract_batch_from_archive(parameters_df_datasets, 'gdppr')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md ##Lookups

# COMMAND ----------

snomed_lookup = spark.table(f'dss_corporate.gdppr_cluster_refset')

# COMMAND ----------

snomed_lookup_clean = (
    snomed_lookup
    .select(f.col("ConceptID").alias("CODE"),f.col("ConceptId_Description").alias("SNOMED_DESCRIPTION"))
    .distinct()
    )

# checking there are no codes that appear > 1 in different clusters that have a different description
display(snomed_lookup_clean
        .groupBy("CODE").count()
        .join(snomed_lookup_clean,on="CODE",how="left")
        .filter(f.col("count")>1)
        )

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

individual_censor_dates_flag

# COMMAND ----------

if individual_censor_dates_flag==True:
    individual_censor_dates = spark.table(individual_censor_dates_table)
else:
    individual_censor_dates = demographics.select("PERSON_ID",f.col("DATE_OF_DEATH").alias("CENSOR_END")).filter(f.col("CENSOR_END").isNotNull())

# COMMAND ----------

gdppr_prepared = (
  gdppr
  .select(f.col('NHS_NUMBER_DEID').alias("PERSON_ID"), 'DATE', 'CODE', 'VALUE1_CONDITION', 'VALUE2_CONDITION')
  .where(f.col('PERSON_ID').isNotNull())  
  .withColumn("DATE", f.when((f.col("DATE") == "1800-01-01") | (f.col("DATE") == "1801-01-01"), None).otherwise(f.col("DATE")))
  .where(f.col('DATE').isNotNull())
  .filter(f.col("DATE")<last_observable_date)
  #filter out records > DATE_OF_DEATH
  .join(individual_censor_dates,on="PERSON_ID",how="left")
  .withColumn("CENSOR_END", f.when(f.col("CENSOR_END").isNull(),last_observable_date).otherwise(f.col("CENSOR_END")))
  .filter(f.col("DATE")<=f.col("CENSOR_END"))
  .drop("CENSOR_END")
)

# COMMAND ----------

if individual_censor_dates_flag==True:
    gdppr_prepared = gdppr_prepared
else:
    gdppr_prepared = gdppr_prepared.filter(f.col("DATE")<study_end_date)

# COMMAND ----------

gdppr_with_lookup = (
    gdppr_prepared
    # .join(snomed_lookup_clean,on="CODE",how="left")
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=gdppr_with_lookup, out_name=f'{proj}_curated_data_gdppr_{algorithm_timestamp}', save_previous=False)