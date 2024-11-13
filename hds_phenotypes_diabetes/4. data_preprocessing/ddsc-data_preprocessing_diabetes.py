# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - Diabetes codes cohort
# MAGIC
# MAGIC **Description** This notebook curates the diabetes codes cohort.
# MAGIC
# MAGIC The following data cleaning/pre-processing has been done:
# MAGIC - code detail removed and distinct diabetes type is found with date
# MAGIC - demographics are used to remove Type 2 codes found in a persons first year from birth (meaning that a persons first Type 2 will be after their DOB + 1 year)
# MAGIC - diabetes records before DOB removed
# MAGIC
# MAGIC Feature enginerring is them performed to establish the following (by diabetes type and primary/secondary/both source):
# MAGIC - If a code type was ever found in a persons history
# MAGIC - The number of codes found across a persons full history (at distinct dates e.g. Type 1 code A and Type 1 code B both found at Date 1 would be considered as one Type 1 code. i.e. it is no. of unique dates with any Type 1 code as opposed to a count of different types of Type 1 codes found)
# MAGIC - The date at which the first code was found in a person's histroy
# MAGIC - The date at which the last code was found in a person's history
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_data_preprocessing_diabetes_long_{algorithm_timestamp}`**
# MAGIC - **`ddsc_data_preprocessing_diabetes_{algorithm_timestamp}`**

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

diabetes_codes = spark.table(f'{dsa}.{proj}_curated_assets_diabetes_{algorithm_timestamp}')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

display(diabetes_codes)

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

 diabetes_cleaned = (
        diabetes_codes
        .drop("code")
        .distinct()
        .join((demographics.select(f.col("PERSON_ID"),"date_of_birth")),how="left",on="PERSON_ID")
        .filter(f.col("DATE")>=f.col("date_of_birth"))
        .withColumn("first_year_birth", f.expr("date_add(date_of_birth, 365)"))
)
        
persons_with_type2_in_first_year = (
    diabetes_cleaned.filter(~((f.col("type") != "Type 2") | (f.col("DATE") >= f.col("first_year_birth"))))
    .select("PERSON_ID").distinct().count()
)

print(f'Number of people who have a Type 2 record in first year from birth: {persons_with_type2_in_first_year}')

# COMMAND ----------

 diabetes_cleaned = (
         diabetes_cleaned
        .filter((f.col("type") != "Type 2") | (f.col("DATE") >= f.col("first_year_birth")))
        .drop("date_of_birth","first_year_birth")
)

# COMMAND ----------

diabetes_wrangled = (

    diabetes_cleaned


    .withColumn("TYPE1_EVER_PRIMARY", f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "SNOMED"), 1).otherwise(0))
    .withColumn("TYPE2_EVER_PRIMARY", f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "SNOMED"), 1).otherwise(0))
    .withColumn("NOS_EVER_PRIMARY", f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "SNOMED"), 1).otherwise(0))
    .withColumn("OTHER_EVER_PRIMARY", f.when((f.col("type") == "Other") & (f.col("coding_system") == "SNOMED"), 1).otherwise(0))

    .withColumn("TYPE1_EVER_SECONDARY", f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "ICD10"), 1).otherwise(0))
    .withColumn("TYPE2_EVER_SECONDARY", f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "ICD10"), 1).otherwise(0))
    .withColumn("NOS_EVER_SECONDARY", f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "ICD10"), 1).otherwise(0))
    .withColumn("OTHER_EVER_SECONDARY", f.when((f.col("type") == "Other") & (f.col("coding_system") == "ICD10"), 1).otherwise(0))

    .groupBy("PERSON_ID").agg(

    (f.max(f.col("TYPE1_EVER_PRIMARY")) > 0).cast("int").alias("TYPE1_EVER_PRIMARY"),
    (f.max(f.col("TYPE2_EVER_PRIMARY")) > 0).cast("int").alias("TYPE2_EVER_PRIMARY"),
    (f.max(f.col("NOS_EVER_PRIMARY")) > 0).cast("int").alias("NOS_EVER_PRIMARY"),
    (f.max(f.col("OTHER_EVER_PRIMARY")) > 0).cast("int").alias("OTHER_EVER_PRIMARY"),
    (f.max(f.col("TYPE1_EVER_SECONDARY")) > 0).cast("int").alias("TYPE1_EVER_SECONDARY"),
    (f.max(f.col("TYPE2_EVER_SECONDARY")) > 0).cast("int").alias("TYPE2_EVER_SECONDARY"),
    (f.max(f.col("NOS_EVER_SECONDARY")) > 0).cast("int").alias("NOS_EVER_SECONDARY"),
    (f.max(f.col("OTHER_EVER_SECONDARY")) > 0).cast("int").alias("OTHER_EVER_SECONDARY"),

    f.count(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "SNOMED"), True)).alias("TYPE1_NO_PRIMARY"),
    f.count(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "SNOMED"), True)).alias("TYPE2_NO_PRIMARY"),
    f.count(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "SNOMED"), True)).alias("NOS_NO_PRIMARY"),
    f.count(f.when((f.col("type") == "Other") & (f.col("coding_system") == "SNOMED"), True)).alias("OTHER_NO_PRIMARY"),
    f.count(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "ICD10"), True)).alias("TYPE1_NO_SECONDARY"),
    f.count(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "ICD10"), True)).alias("TYPE2_NO_SECONDARY"),
    f.count(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "ICD10"), True)).alias("NOS_NO_SECONDARY"),
    f.count(f.when((f.col("type") == "Other") & (f.col("coding_system") == "ICD10"), True)).alias("OTHER_NO_SECONDARY"),

    f.min(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("TYPE1_DATE_FIRST_PRIMARY"),
    f.min(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("TYPE2_DATE_FIRST_PRIMARY"),
    f.min(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("NOS_DATE_FIRST_PRIMARY"),
    f.min(f.when((f.col("type") == "Other") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("OTHER_DATE_FIRST_PRIMARY"),
    f.min(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("TYPE1_DATE_FIRST_SECONDARY"),
    f.min(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("TYPE2_DATE_FIRST_SECONDARY"),
    f.min(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("NOS_DATE_FIRST_SECONDARY"),
    f.min(f.when((f.col("type") == "Other") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("OTHER_DATE_FIRST_SECONDARY"),

    f.max(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("TYPE1_DATE_LAST_PRIMARY"),
    f.max(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("TYPE2_DATE_LAST_PRIMARY"),
    f.max(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("NOS_DATE_LAST_PRIMARY"),
    f.max(f.when((f.col("type") == "Other") & (f.col("coding_system") == "SNOMED"), f.col("DATE"))).alias("OTHER_DATE_LAST_PRIMARY"),
    f.max(f.when((f.col("type") == "Type 1") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("TYPE1_DATE_LAST_SECONDARY"),
    f.max(f.when((f.col("type") == "Type 2") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("TYPE2_DATE_LAST_SECONDARY"),
    f.max(f.when((f.col("type") == "Diabetes NOS") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("NOS_DATE_LAST_SECONDARY"),
    f.max(f.when((f.col("type") == "Other") & (f.col("coding_system") == "ICD10"), f.col("DATE"))).alias("OTHER_DATE_LAST_SECONDARY")
)

    .withColumn("TYPE1_EVER", f.when((f.col("TYPE1_EVER_PRIMARY") == 1) | (f.col("TYPE1_EVER_SECONDARY") == 1), 1).otherwise(0))
    .withColumn("TYPE2_EVER", f.when((f.col("TYPE2_EVER_PRIMARY") == 1) | (f.col("TYPE2_EVER_SECONDARY") == 1), 1).otherwise(0))
    .withColumn("NOS_EVER", f.when((f.col("NOS_EVER_PRIMARY") == 1) | (f.col("NOS_EVER_SECONDARY") == 1), 1).otherwise(0))
    .withColumn("OTHER_EVER", f.when((f.col("OTHER_EVER_PRIMARY") == 1) | (f.col("OTHER_EVER_SECONDARY") == 1), 1).otherwise(0))

    .withColumn("TYPE1_NO", f.col("TYPE1_NO_PRIMARY") + f.col("TYPE1_NO_SECONDARY"))
    .withColumn("TYPE2_NO", f.col("TYPE2_NO_PRIMARY") + f.col("TYPE2_NO_SECONDARY"))
    .withColumn("NOS_NO", f.col("NOS_NO_PRIMARY") + f.col("NOS_NO_SECONDARY"))
    .withColumn("OTHER_NO", f.col("OTHER_NO_PRIMARY") + f.col("OTHER_NO_SECONDARY"))

    .withColumn("TYPE1_DATE_FIRST", f.least(f.col("TYPE1_DATE_FIRST_PRIMARY"), f.col("TYPE1_DATE_FIRST_SECONDARY")))
    .withColumn("TYPE2_DATE_FIRST", f.least(f.col("TYPE2_DATE_FIRST_PRIMARY"), f.col("TYPE2_DATE_FIRST_SECONDARY")))
    .withColumn("NOS_DATE_FIRST", f.least(f.col("NOS_DATE_FIRST_PRIMARY"), f.col("NOS_DATE_FIRST_SECONDARY")))
    .withColumn("OTHER_DATE_FIRST", f.least(f.col("OTHER_DATE_FIRST_PRIMARY"), f.col("OTHER_DATE_FIRST_SECONDARY")))

    .withColumn("TYPE1_DATE_LAST", f.greatest(f.col("TYPE1_DATE_LAST_PRIMARY"), f.col("TYPE1_DATE_LAST_SECONDARY")))
    .withColumn("TYPE2_DATE_LAST", f.greatest(f.col("TYPE2_DATE_LAST_PRIMARY"), f.col("TYPE2_DATE_LAST_SECONDARY")))
    .withColumn("NOS_DATE_LAST", f.greatest(f.col("NOS_DATE_LAST_PRIMARY"), f.col("NOS_DATE_LAST_SECONDARY")))
    .withColumn("OTHER_DATE_LAST", f.greatest(f.col("OTHER_DATE_LAST_PRIMARY"), f.col("OTHER_DATE_LAST_SECONDARY")))

    .withColumn("min_diabetes_date", f.least(f.col("TYPE1_DATE_FIRST"),
                                                f.col("TYPE2_DATE_FIRST"),
                                                f.col("NOS_DATE_FIRST"),
                                                f.col("OTHER_DATE_FIRST"))
                )
    .withColumn("max_diabetes_date", f.greatest(f.col("TYPE1_DATE_LAST"),
                                                f.col("TYPE2_DATE_LAST"),
                                                f.col("NOS_DATE_LAST"),
                                                f.col("OTHER_DATE_LAST"))
                )

    # flag for if in diabetes codes cohort
    .withColumn("flag_diabetes_cohort",f.lit(1))

)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=diabetes_cleaned, out_name=f'{proj}_data_preprocessing_diabetes_long_{algorithm_timestamp}', save_previous=False)

# COMMAND ----------

save_table(df=diabetes_wrangled, out_name=f'{proj}_data_preprocessing_diabetes_{algorithm_timestamp}', save_previous=False)