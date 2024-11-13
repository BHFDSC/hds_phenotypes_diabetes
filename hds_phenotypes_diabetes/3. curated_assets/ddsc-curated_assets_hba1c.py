# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Assets - HbA1c
# MAGIC
# MAGIC **Description** This notebook curates a table detailing an individual's complete history of HbA1c readings. HbA1c data is derived from GDPPR.
# MAGIC
# MAGIC The following **data cleaning** has been applied:
# MAGIC - records that are outwith realistic lower and upper limits removed (ie remove HbA1c readings <10 and >200)
# MAGIC - where a person has more than one reading on the same day, the mean is imputed
# MAGIC - hba1c groups have been added: If HbA1c >=48 mmol/mol then category high; else not high
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_assets_hba1c_{algorithm_timestamp}`** 

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

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md # Codelists

# COMMAND ----------

codelist_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/hba1c.csv'
paths = [codelist_path]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

codelist_hba1c = codelist.select(f.col("code").alias("CODE"))

codelist_hba1c = codelist_hba1c.withColumn("CODE", f.col("CODE").cast("string"))

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

gdppr_prepared = spark.table(f'{dsa}.{proj}_curated_data_gdppr_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md # Curate

# COMMAND ----------

# gdppr_hba1c = codelist_hba1c.join(gdppr_prepared,on="CODE",how="inner")

unique_codes_list = [row['CODE'] for row in codelist_hba1c.select("CODE").distinct().collect()]
gdppr_hba1c = gdppr_prepared.filter(f.col("CODE").isin(unique_codes_list))

# COMMAND ----------

# DBTITLE 1,Check: no VALUE2_CONDITION values
display(gdppr_hba1c.filter(f.col("VALUE2_CONDITION").isNotNull()).count())

# COMMAND ----------

hba1c_cohort = (
  gdppr_hba1c.drop("VALUE2_CONDITION")
  .withColumnRenamed("VALUE1_CONDITION","val")
  .withColumn('val', f.round(f.col('val'), 0))
  .filter(f.col('val').isNotNull())
  .withColumnRenamed("val","hba1c")
  .distinct()
)

# COMMAND ----------

# DBTITLE 1,Check: % of records outside lower and upper limits
# lower = hba1c_cohort.filter(f.col("hba1c")<10).count()
# upper = hba1c_cohort.filter(f.col("hba1c")>200).count()

# print(f'Number of records readings <10: {lower}')
# print(f'Number of records readings >200: {upper}')

# COMMAND ----------

hba1c_cohort = (
    hba1c_cohort
    #remove records that are outwith limits (ie remove <10 and >200)
    .filter(f.col("hba1c")>=10)
    .filter(f.col("hba1c")<=200)
)

# COMMAND ----------

# DBTITLE 1,Check: % of records with ties
# ties = (
#     hba1c_cohort
#     .groupBy("PERSON_ID", "DATE").count()
#     .filter(f.col("count")>1)
#     .count()

# )

# window_spec = Window.partitionBy("PERSON_ID", "DATE")
# tmp = hba1c_cohort.withColumn("count", f.count("*").over(window_spec))
# ties = tmp.filter(f.col("count") > 1).select("PERSON_ID", "DATE").distinct().count()

# COMMAND ----------

hba1c_cohort = (
    hba1c_cohort
    #ties - if more than one record on same day then take mean
    .groupBy("PERSON_ID", "DATE").agg(f.mean("hba1c").alias("hba1c"))
    .withColumn('val_group',f.when(f.col("hba1c")>=48,"High").otherwise("Not High"))
    .select("PERSON_ID","DATE","hba1c",f.col("val_group").alias("hba1c_group"))
    .distinct()
)

# COMMAND ----------

# distinct_persons_date = (hba1c_cohort.select("PERSON_ID", "DATE").distinct().count())
# print(f'Number of ties as a %: {ties}, ({round(ties/distinct_persons_date*100,4)}%)')

# COMMAND ----------

# MAGIC %md # Save

# COMMAND ----------

hba1c_cohort = hba1c_cohort.repartition(1536)
save_table(df=hba1c_cohort, out_name=f'{proj}_curated_assets_hba1c_{algorithm_timestamp}', save_previous=False)

# COMMAND ----------

hba1c_cohort = spark.table(f'{dsa}.{proj}_curated_assets_hba1c_{algorithm_timestamp}')
display(hba1c_cohort.agg({"DATE": "min"}).collect()[0][0])