# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Assets - Insulin
# MAGIC
# MAGIC **Description** This notebook curates a table detailing an individual's complete history of insulin prescriptions. Prescription data is derived from PMEDS.
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_assets_medications_insulin_{algorithm_timestamp}`** 

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

import os

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md # Codelists

# COMMAND ----------

medications_insulin_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/medications_insulin.csv'
medications_insulin_pandas_df = pd.read_csv(medications_insulin_path, keep_default_na=False)
medications_insulin_spark_df = spark.createDataFrame(medications_insulin_pandas_df)

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

pmeds_prepared = spark.table(f'{dsa}.{proj}_curated_data_pmeds_{algorithm_timestamp}')

# COMMAND ----------

bnf_lookup = (
    spark.table(f'dss_corporate.bnf_code_information')
    .select(f.col("BNF_Presentation_Code").alias("CODE"),
             f.col("BNF_Presentation").alias("CODE_DESCRIPTION"),
             f.col("BNF_Chemical_Substance_Code").alias("CODE_PARENT"),
             f.col("BNF_Chemical_Substance").alias("CODE_PARENT_DESCRIPTION"),
             )
              )

# COMMAND ----------

# MAGIC %md # Curate

# COMMAND ----------

unique_codes_list = [row['code'] for row in medications_insulin_spark_df.select("code").distinct().collect()]

insulin_cohort_pmeds = (
    pmeds_prepared
    .filter(f.col("CODE_PARENT").isin(unique_codes_list))
    .join((bnf_lookup.select("CODE","CODE_DESCRIPTION")), on="CODE",how="left")
    .join((bnf_lookup.select("CODE_PARENT","CODE_PARENT_DESCRIPTION")), on="CODE_PARENT",how="left")
    .select("PERSON_ID","DATE","CODE_PARENT",
            "CODE_PARENT_DESCRIPTION",
            "CODE",
            "CODE_DESCRIPTION",
            "PrescribedQuantity",
            "PrescribedMedicineStrength"
            )
    .distinct()
)

# COMMAND ----------

# MAGIC %md # Save

# COMMAND ----------

save_table(df=insulin_cohort_pmeds, out_name=f'{proj}_curated_assets_medications_insulin_{algorithm_timestamp}', save_previous=False)