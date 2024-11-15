# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Data - PMEDs
# MAGIC
# MAGIC **Description** This notebook curates a prepared version of PMEDs for use in all subsequent notebooks.
# MAGIC
# MAGIC The following data cleaning has been done:
# MAGIC - PMEDs filtered to the archived_on batch defined in paramters
# MAGIC - Wrangled into long format (codes per row)
# MAGIC - Any records dated 1800-01-01 or 1801-01-01 removed
# MAGIC - Records > than the last observable date removed
# MAGIC - Individual censor dates applied and for those who have died, records > date of death removed
# MAGIC - Null dates and PERSON_ID removed
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_data_pmeds_{algorithm_timestamp}`**

# COMMAND ----------

# MAGIC %md # Setup

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

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %md # Parameters

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-last_observable_date"

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %md # Curate

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

if individual_censor_dates_flag==True:
    individual_censor_dates = spark.table(individual_censor_dates_table)
else:
    individual_censor_dates = demographics.select("PERSON_ID",f.col("DATE_OF_DEATH").alias("CENSOR_END")).filter(f.col("CENSOR_END").isNotNull())

# COMMAND ----------

pmeds = extract_batch_from_archive(parameters_df_datasets, 'pmeds')

# COMMAND ----------

pmeds_prepared = (pmeds
  .select(['Person_ID_DEID', 'ProcessingPeriodDate', 'PrescribedBNFCode',
            "PrescribedQuantity","PrescribedMedicineStrength"])
  .withColumnRenamed('Person_ID_DEID', 'PERSON_ID')
  .withColumnRenamed('ProcessingPeriodDate', 'DATE')
  .withColumnRenamed('PrescribedBNFCode', 'CODE')
  .withColumn("DATE", f.when((f.col("DATE") == "1800-01-01") | (f.col("DATE") == "1801-01-01"), None).otherwise(f.col("DATE")))
  .withColumn("CODE_PARENT", f.substring(f.col("CODE"), 1, 9))
  .filter(f.col("DATE")<last_observable_date)
  #filter out records > DATE_OF_DEATH
  .join(individual_censor_dates,on="PERSON_ID",how="left")
  .withColumn("CENSOR_END", f.when(f.col("CENSOR_END").isNull(),last_observable_date).otherwise(f.col("CENSOR_END")))
  .filter(f.col("DATE")<=f.col("CENSOR_END"))
  .drop("CENSOR_END")
  .withColumn(
    "DATE",
    f.when(f.month("DATE") == 1, f.expr("concat(year(DATE),'-01-16')").cast("date"))
    .when(f.month("DATE") == 2, f.expr("concat(year(DATE),'-02-15')").cast("date"))
    .when(f.month("DATE") == 3, f.expr("concat(year(DATE),'-03-16')").cast("date"))
    .when(f.month("DATE") == 4, f.expr("concat(year(DATE),'-04-16')").cast("date"))
    .when(f.month("DATE") == 5, f.expr("concat(year(DATE),'-05-16')").cast("date"))
    .when(f.month("DATE") == 6, f.expr("concat(year(DATE),'-06-16')").cast("date"))
    .when(f.month("DATE") == 7, f.expr("concat(year(DATE),'-07-16')").cast("date"))
    .when(f.month("DATE") == 8, f.expr("concat(year(DATE),'-08-16')").cast("date"))
    .when(f.month("DATE") == 9, f.expr("concat(year(DATE),'-09-16')").cast("date"))
    .when(f.month("DATE") == 10, f.expr("concat(year(DATE),'-10-16')").cast("date"))
    .when(f.month("DATE") == 11, f.expr("concat(year(DATE),'-11-16')").cast("date"))
    .otherwise(f.expr("concat(year(DATE),'-12-16')").cast("date"))
)
)

# COMMAND ----------

if individual_censor_dates_flag==True:
    pmeds_prepared = pmeds_prepared
else:
    pmeds_prepared = pmeds_prepared.filter(f.col("DATE")<study_end_date)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=pmeds_prepared, out_name=f'{proj}_curated_data_pmeds_{algorithm_timestamp}', save_previous=False)