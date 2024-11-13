# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - Metformin cohort
# MAGIC
# MAGIC **Description** This notebook curates the Metformin cohort.
# MAGIC
# MAGIC The following data cleaning/pre-processing has been done:
# MAGIC - records before DOB removed
# MAGIC
# MAGIC Feature enginerring is them performed to establish the following:
# MAGIC - If Metformin medication has ever been prescribed in a person's history
# MAGIC - The number of Metformin prescriptions in a person's history
# MAGIC - The date at which the first Metformin prescription was found in a person's history
# MAGIC - The date at which the last Metformin prescription was found in a person's history
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_data_preprocessing_medications_metformin_{algorithm_timestamp}`**

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

Metformin = spark.table(f'{dsa}.{proj}_curated_assets_medications_metformin_{algorithm_timestamp}')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

Metformin = (
        Metformin
        .join((demographics.select(f.col("PERSON_ID"),"date_of_birth")),how="left",on="PERSON_ID")
        .filter(f.col("DATE")>=f.col("date_of_birth"))
        .drop("date_of_birth")
        .select("PERSON_ID","DATE")
        .distinct()
)

# COMMAND ----------

metformin_no = (
    Metformin.select("PERSON_ID","DATE").groupBy("PERSON_ID").count().withColumnRenamed("count","METFORMIN_NO")
    .withColumn("METFORMIN_EVER",f.lit(1))
)

# Last Date
_win_last = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE').desc()))

metformin_date_last = (
    Metformin.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win_last))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","METFORMIN_DATE_LAST")
)

# First Date
_win = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE')))

metformin_date_first = (
    Metformin.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","METFORMIN_DATE_FIRST")
)

metformin_final = (
    metformin_no
    .join(metformin_date_first,on="PERSON_ID",how="outer")
    .join(metformin_date_last,on="PERSON_ID",how="outer")

    .select("PERSON_ID",
            "METFORMIN_EVER","METFORMIN_NO","METFORMIN_DATE_FIRST","METFORMIN_DATE_LAST"
            )
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=metformin_final, out_name=f'{proj}_data_preprocessing_medications_metformin_{algorithm_timestamp}', save_previous=False)