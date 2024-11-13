# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - Date of Diagnosis
# MAGIC
# MAGIC **Description** This notebook curates the date of diagnosis, defined as the first diabetes code (of any type) or the first high HbA1c (of at least 2 consecutive records that fall within 2 years of each other) if more than a year before first diabetes code.
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_data_preprocessing_date_of_diagnosis_{algorithm_timestamp}`**

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

diabetes = spark.table(f'{dsa}.{proj}_data_preprocessing_diabetes_{algorithm_timestamp}').select("PERSON_ID","min_diabetes_date")

# COMMAND ----------

hba1c = spark.table(f'{dsa}.{proj}_data_preprocessing_hba1c_{algorithm_timestamp}').select("PERSON_ID","HBA1C_HIGH_DATE_FIRST")

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

date_of_diagnosis = (
    diabetes.join(hba1c,on="PERSON_ID",how="full")
    .withColumn("one_year_ago", f.date_sub(f.col("min_diabetes_date"), 365))
    .withColumn(
    "date_of_diagnosis",
    f.when(
            f.col("HBA1C_HIGH_DATE_FIRST").isNotNull() &
            (f.col("HBA1C_HIGH_DATE_FIRST") < f.col("one_year_ago")),
            f.col("HBA1C_HIGH_DATE_FIRST")
    ).otherwise(
        f.when(
            f.col("min_diabetes_date").isNotNull(),
            f.col("min_diabetes_date")
        ).otherwise(
            f.col("HBA1C_HIGH_DATE_FIRST")
        )
    )
)
    .select("PERSON_ID","date_of_diagnosis")
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=date_of_diagnosis, out_name=f'{proj}_data_preprocessing_date_of_diagnosis_{algorithm_timestamp}', save_previous=False)