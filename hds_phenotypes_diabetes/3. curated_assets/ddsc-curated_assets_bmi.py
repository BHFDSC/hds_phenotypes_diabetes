# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Assets - BMI
# MAGIC
# MAGIC **Description** This notebook curates a table detailing an individual's complete history of BMI readings. BMI data is derived from GDPPR (in the form of markers and categories) and HES APC (in the form of categories).
# MAGIC
# MAGIC The following **data cleaning** has been applied:
# MAGIC - records that are outwith realistic lower and upper limits removed (ie remove BMI readings <10 and >270)
# MAGIC - where a person has more than one marker on the same day, the mean is imputed
# MAGIC
# MAGIC BMI groups are derived from markers as follows:
# MAGIC - Underweight: BMI <18.5
# MAGIC - Normal: >=18.5 and <25
# MAGIC - Overweight: >=25 and <30
# MAGIC - Obese: >=30
# MAGIC <br>
# MAGIC <br>
# MAGIC
# MAGIC Ties between the markers and category groups:
# MAGIC - If a marker is included in the tie then the record with the marker is selected over any other categories
# MAGIC - Thereafter, the higest BMI category is selected: e.g. Obese > Overweight > Normal > Underweight
# MAGIC
# MAGIC Update to above: categories are now not included and markers are prioritised only. There will be no ties at a category level as all categories are mapped from the markers (and where there was ties the average was taken).
# MAGIC
# MAGIC **Author(s)** Tom Bolton, Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_assets_bmi_{algorithm_timestamp}`**

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

# DBTITLE 1,Functions
# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # Codelists

# COMMAND ----------

# codelists
codelist_path_bmi_markers = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/bmi_markers.csv'
codelist_path_bmi_categories = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/1. codelists/bmi_categories.csv'
paths = [codelist_path_bmi_markers,codelist_path_bmi_categories]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

# COMMAND ----------

codelist_bmi_markers = (codelist.filter(f.col('group').isin(["bmi_markers"])).select(f.col("code").alias("CODE")).distinct())


codelist_bmi_categories_icd = (codelist.filter(~f.col('group').isin(["bmi_markers"])).filter(f.col("terminology").isin(["ICD10"]))
                        .select(f.col("code").alias("CODE"),'group').distinct())

codelist_bmi_categories_snomed = (codelist.filter(~f.col('group').isin(["bmi_markers"])).filter(f.col("terminology").isin(["SNOMED"]))
                        .select(f.col("code").alias("CODE"),'group').distinct())



# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

gdppr_prepared = spark.table(f'{dsa}.{proj}_curated_data_gdppr_{algorithm_timestamp}')
hes_apc = spark.table(f'{dsa}.{proj}_curated_data_hes_apc_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Curate

# COMMAND ----------

# MAGIC %md
# MAGIC ## Markers

# COMMAND ----------

# gdppr_bmi = codelist_bmi_markers.join(gdppr_prepared,on="CODE",how="inner")

unique_codes_list = [row['CODE'] for row in codelist_bmi_markers.select("CODE").distinct().collect()]
gdppr_bmi = gdppr_prepared.filter(f.col("CODE").isin(unique_codes_list))

# COMMAND ----------

# DBTITLE 1,Check: no VALUE2_CONDITION values
display(gdppr_bmi.filter(f.col("VALUE2_CONDITION").isNotNull()))

# COMMAND ----------

bmi_cohort = (
  gdppr_bmi.drop("VALUE2_CONDITION")
  .withColumnRenamed("VALUE1_CONDITION","val")
  .filter(f.col('val').isNotNull())
  .withColumn('val', f.round(f.col('val'), 0))
  .select("PERSON_ID","DATE",f.col("val").alias("bmi"))
  .distinct()
)

# COMMAND ----------

min_bmi = bmi_cohort.select(f.min("bmi")).collect()[0][0]
max_bmi = bmi_cohort.select(f.max("bmi")).collect()[0][0]

# Print the results
print(f"Minimum BMI: {min_bmi}")
print(f"Maximum BMI: {max_bmi}")

# COMMAND ----------

bmi_cohort = (
    bmi_cohort
    .filter(f.col("bmi")>=10)
    .filter(f.col("bmi")<=270)
)

# COMMAND ----------

underweight_threshold = 18.5
normal_weight_threshold = 25
overweight_threshold = 30

bmi_cohort = (
    bmi_cohort
    #ties - if more than one record on same day then take mean
    .groupBy("PERSON_ID", "DATE").agg(f.mean("bmi").alias("bmi"))
    #categories
    .withColumn("bmi_group",
              f.when(f.col("bmi") < underweight_threshold, "Underweight")
              .when((f.col("bmi") >= underweight_threshold) & (f.col("bmi") < normal_weight_threshold), "Normal")
              .when((f.col("bmi") >= normal_weight_threshold) & (f.col("bmi") < overweight_threshold), "Overweight")
              .otherwise("Obese"))
    .orderBy("PERSON_ID","DATE","bmi")
    .withColumn("source",f.lit("gdppr"))
    .distinct()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categories

# COMMAND ----------

# MAGIC %md
# MAGIC Note that categories will no longer be used but code will remain here.

# COMMAND ----------

# MAGIC %md
# MAGIC ### GDPPR

# COMMAND ----------

# gdppr_bmi_groups = codelist_bmi_categories_snomed.join(gdppr_prepared,on="CODE",how="inner")


# bmi_groups_cohort_snomed = (
#     gdppr_bmi_groups
#     .withColumn("bmi",f.lit(None))
#     .withColumn("source",f.lit("gdppr"))
#     .select("PERSON_ID","DATE","bmi",f.col('group').alias("bmi_group"),"source")
#     .withColumn("bmi_group", f.when(f.col("bmi")=="obesity","Obese")
#                 .when(f.col("bmi")=="overweight","Overweight")
#                 .when(f.col("bmi")=="underweight","Underweight")
#                 .otherwise("Normal")
#                 )
#    .distinct()
#    .orderBy("PERSON_ID","DATE")
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ### HES APC

# COMMAND ----------

# hes_apc_bmi_groups = codelist_bmi_categories_icd.join(hes_apc,on="CODE",how="inner")

# bmi_groups_cohort_icd = (
#     hes_apc_bmi_groups
#     .withColumn("bmi",f.lit(None))
#     .withColumn("source",f.lit("hes_apc"))
#     .select("PERSON_ID",f.col("EPISTART").alias("DATE"),"bmi",f.col('group').alias("bmi_group"),"source")
#     .withColumn("bmi_group", f.when(f.col("bmi")=="obesity","Obese")
#                 .when(f.col("bmi")=="overweight","Overweight")
#                 .when(f.col("bmi")=="underweight","Underweight")
#                 .otherwise("Normal")
#                 )
#    .distinct()
#    .orderBy("PERSON_ID","DATE")
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine

# COMMAND ----------

bmi_groups_all = (
    bmi_cohort
    # .union(bmi_groups_cohort_snomed)
    # .union(bmi_groups_cohort_icd)
    .orderBy("PERSON_ID","DATE","bmi")
    .filter(f.col("PERSON_ID").isNotNull())
    .distinct()
)

# this version has no ties removed

# COMMAND ----------

display(bmi_groups_all)

# COMMAND ----------

# display(bmi_groups_all.select("PERSON_ID","DATE","bmi_group").distinct().groupBy("PERSON_ID","DATE").count().filter(f.col("count")>1).count())

# COMMAND ----------

# display(bmi_groups_all.select("PERSON_ID","DATE","bmi_group").distinct().groupBy("PERSON_ID","DATE").count().filter(f.col("count")==1).count())

# COMMAND ----------

bmi_groups_all = (
    bmi_groups_all
    .withColumn("priority", 
                   f.when(f.col("bmi").isNotNull(), 1).otherwise(
                      f.when(f.col("bmi_group") == "Obese", 2)
                      .when(f.col("bmi_group") == "Overweight", 3)
                      .when(f.col("bmi_group") == "Normal", 4)
                      .when(f.col("bmi_group") == "Underweight", 5)
                   ))
    
)

windowSpec = Window.partitionBy("PERSON_ID", "DATE").orderBy("priority")

bmi_groups_all = (
    bmi_groups_all.withColumn("rn", f.row_number().over(windowSpec)).filter(f.col("rn") == 1).drop("rn","priority")
)

# df = bmi_groups_all.select("PERSON_ID","DATE","bmi_group").distinct().groupBy("PERSON_ID","DATE").count().filter(f.col("count")>1).join(bmi_groups_all,on=["PERSON_ID","DATE"],how="left").withColumn("rn", f.row_number().over(windowSpec))
# display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Save

# COMMAND ----------

save_table(df=bmi_groups_all, out_name=f'{proj}_curated_assets_bmi_{algorithm_timestamp}', save_previous=False)