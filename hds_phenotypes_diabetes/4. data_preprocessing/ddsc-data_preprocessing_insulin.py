# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - Insulin cohort
# MAGIC
# MAGIC **Description** This notebook curates the Insulin cohort.
# MAGIC
# MAGIC The following data cleaning/pre-processing has been done:
# MAGIC - records before DOB removed
# MAGIC
# MAGIC Feature enginerring is them performed to establish the following:
# MAGIC - If Insulin has ever been prescribed in a person's history
# MAGIC - The number of Insulin prescriptions in a person's history
# MAGIC - The date at which the first Insulin prescription was found in a person's history
# MAGIC - The date at which the last Insulin prescription was found in a person's history
# MAGIC
# MAGIC Additionally, for inclusion in the 'at least 6 months of Insulin' cohort
# MAGIC - If a person has at least 6 months of Insulin prescribing data available i.e. the difference between the first and last Insulin prescription found in a peron's history is > 6 months
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_data_preprocessing_medications_insulin_{algorithm_timestamp}`**

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

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

insulin = spark.table(f'{dsa}.{proj}_curated_assets_medications_insulin_{algorithm_timestamp}')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

display(insulin)

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

insulin = (
        insulin
        .join((demographics.select(f.col("PERSON_ID"),"date_of_birth")),how="left",on="PERSON_ID")
        .filter(f.col("DATE")>=f.col("date_of_birth"))
        .drop("date_of_birth")
        .select("PERSON_ID","DATE")
        .distinct()
)

# COMMAND ----------

insulin_no = (
    insulin.select("PERSON_ID","DATE").groupBy("PERSON_ID").count().withColumnRenamed("count","INSULIN_NO")
    .withColumn("INSULIN_EVER",f.lit(1))
)

# Last Date
_win_last = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE').desc()))

insulin_date_last = (
    insulin.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win_last))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","INSULIN_DATE_LAST")
)

# First Date
_win = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE')))

insulin_date_first = (
    insulin.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","INSULIN_DATE_FIRST")
)

insulin_final = (
    insulin_no
    .join(insulin_date_first,on="PERSON_ID",how="outer")
    .join(insulin_date_last,on="PERSON_ID",how="outer")

    .select("PERSON_ID",
            "INSULIN_EVER","INSULIN_NO","INSULIN_DATE_FIRST","INSULIN_DATE_LAST"
            )
    
    .withColumn("MONTH_DIFF", f.months_between(f.col("INSULIN_DATE_LAST"), f.col("INSULIN_DATE_FIRST")))
    .withColumn("INSULIN_6_MONTHS", f.when(f.col("MONTH_DIFF") > 6, 1).otherwise(0))
    .drop("MONTH_DIFF")
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=insulin_final, out_name=f'{proj}_data_preprocessing_medications_insulin_{algorithm_timestamp}', save_previous=False)