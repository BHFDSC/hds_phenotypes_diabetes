# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Assets - Diabetes codes from primary and seconadary care
# MAGIC
# MAGIC **Description** This notebook curates a table detailing an individual's complete history of diabetes codes (and type) from GDPPR and HES APC. Code level detail.
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_assets_diabetes_{algorithm_timestamp}`**

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

import os

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md # Codelists

# COMMAND ----------

diabetes_type1_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/diabetes_type1.csv'
diabetes_type2_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/diabetes_type2.csv'
diabetes_other_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/diabetes_other.csv'
diabetes_nos_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/diabetes_nos.csv'

paths = [diabetes_type1_path, diabetes_type2_path, diabetes_other_path, diabetes_nos_path]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    diabetes_codelists = spark_dfs[0]
    for df in spark_dfs[1:]:
        diabetes_codelists = diabetes_codelists.union(df)

diabetes_all_codes_descriptions_matrix =  diabetes_codelists

# COMMAND ----------

diabetes_all_codes_descriptions_matrix = diabetes_all_codes_descriptions_matrix.select("code",f.col("terminology").alias("coding_system"),
                                                                                       f.col("diabetes_type").alias("type")
)


# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

gdppr_prepared = spark.table(f'{dsa}.{proj}_curated_data_gdppr_{algorithm_timestamp}')
hes_apc = spark.table(f'{dsa}.{proj}_curated_data_hes_apc_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

hes_apc_diabetes = (
    diabetes_all_codes_descriptions_matrix.join(hes_apc,on="CODE",how="inner")
    .select("PERSON_ID",f.col("EPISTART").alias("DATE"),"type","code","coding_system")
    
)

gdppr_diabetes = (
    diabetes_all_codes_descriptions_matrix.join(gdppr_prepared,on="CODE",how="inner")
    .select("PERSON_ID",f.col("DATE"),"type","code","coding_system")
    
)

# COMMAND ----------

diabetes = (
    gdppr_diabetes
    .union(hes_apc_diabetes)
    .distinct()
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=diabetes, out_name=f'{proj}_curated_assets_diabetes_{algorithm_timestamp}', save_previous=False)