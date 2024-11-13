# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - BMI cohort
# MAGIC
# MAGIC **Description** This notebook curates the BMI cohort.
# MAGIC
# MAGIC The following data cleaning/pre-processing has been done:
# MAGIC - records before DOB removed
# MAGIC
# MAGIC Feature enginerring is them performed to establish the following:
# MAGIC - If an Obese code was ever found in a persons history
# MAGIC - If an Overweight/Obese code was ever found in a persons history
# MAGIC - The number of Obese codes found across a persons full history
# MAGIC - The number of Overweight/Obese codes found across a persons full history
# MAGIC - The date at which the first Obese code was found in a person's history
# MAGIC - The date at which the first Overweight/Obese code was found in a person's history
# MAGIC - The date at which the last Overweight/Obese code was found in a person's history
# MAGIC - The BMI category that a person was more recently classified as
# MAGIC - The BMI marker that a person was more recently classified as
# MAGIC - The BMI category that a person was classified as at their date of diagnosis (the record closest to diagnosis in either direction)
# MAGIC - The BMI marker that a person was classified as at their date of diagnosis (the record closest to diagnosis in either direction)
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_data_preprocessing_bmi_{algorithm_timestamp}`**

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

bmi = spark.table(f'{dsa}.{proj}_curated_assets_bmi_{algorithm_timestamp}')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

date_of_diagnosis = spark.table(f'{dsa}.{proj}_data_preprocessing_date_of_diagnosis_{algorithm_timestamp}')

# COMMAND ----------

display(bmi.groupBy("PERSON_ID","DATE").count().filter(f.col("count")>1).count())

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

bmi = (
        bmi
        .join((demographics.select(f.col("PERSON_ID"),"date_of_birth")),how="left",on="PERSON_ID")
        .filter(f.col("DATE")>=f.col("date_of_birth"))
        .drop("date_of_birth")
)

# COMMAND ----------

# MAGIC %md ## Obese

# COMMAND ----------

# Obese
obese_bmi = (bmi.filter(f.col("bmi_group")=="Obese").drop("source").distinct())

# No of High Records in history
obese_bmi_no = (
    obese_bmi.select("PERSON_ID","DATE").groupBy("PERSON_ID").count().withColumnRenamed("count","BMI_OBESE_NO")
    .withColumn("BMI_OBESE_EVER",f.lit(1))
)

# Last Date
_win_last = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE').desc()))

obese_bmi_date_last = (
    obese_bmi.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win_last))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","BMI_OBESE_DATE_LAST")
)

# First Date
_win = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE')))

obese_bmi_date_first = (
    obese_bmi.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","BMI_OBESE_DATE_FIRST")
)

# COMMAND ----------

# Obese or Overweight
obese_overweight_bmi = (bmi.filter(f.col("bmi_group").isin('Obese', 'Overweight')).drop("source").distinct())

# No of High Records in history
obese_overweight_bmi_no = (
    obese_overweight_bmi.select("PERSON_ID","DATE").groupBy("PERSON_ID").count().withColumnRenamed("count","BMI_OBESE_OVERWEIGHT_NO")
    .withColumn("BMI_OBESE_OVERWEIGHT_EVER",f.lit(1))
)

# Last Date
_win_last = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE').desc()))

obese_overweight_bmi_date_last = (
    obese_overweight_bmi.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win_last))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","BMI_OBESE_OVERWEIGHT_DATE_LAST")
)

# First Date
_win = (Window.partitionBy('PERSON_ID').orderBy(f.col('DATE')))

obese_overweight_bmi_date_first = (
    obese_overweight_bmi.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","BMI_OBESE_OVERWEIGHT_DATE_FIRST")
)

# COMMAND ----------

# MAGIC %md ## Overweight/Obese

# COMMAND ----------

# MAGIC %md ##Current and closest to Diagnosis

# COMMAND ----------

# Closest BMI marker to date of diagnosis
tmp = (
    bmi.filter(f.col("bmi").isNotNull())
    .join(date_of_diagnosis,on="PERSON_ID",how="left")
    .withColumn("difference", f.abs(f.datediff(f.col("DATE"), f.col("date_of_diagnosis"))))
)

window_spec = Window.partitionBy("PERSON_ID").orderBy("difference")

diagnosis_marker = (
    tmp.withColumn("row_number", f.row_number().over(window_spec))
    .filter(f.col("row_number") == 1).drop("difference", "row_number","date_of_diagnosis")
    .select("PERSON_ID",f.col("bmi").alias("BMI_DATE_DIAGNOSIS_MARKER"))
)

# Closest BMI category to date of diagnosis
tmp = (
    bmi
    .join(date_of_diagnosis,on="PERSON_ID",how="left")
    .withColumn("difference", f.abs(f.datediff(f.col("DATE"), f.col("date_of_diagnosis"))))
)

window_spec = Window.partitionBy("PERSON_ID").orderBy("difference")

diagnosis_category = (
    tmp.withColumn("row_number", f.row_number().over(window_spec))
    .filter(f.col("row_number") == 1).drop("difference", "row_number","date_of_diagnosis")
    .select("PERSON_ID",f.col("bmi_group").alias("BMI_DATE_DIAGNOSIS_CATEGORY"))
)


# COMMAND ----------

# Most recent BMI marker to date of diagnosis
tmp = (
    bmi.filter(f.col("bmi").isNotNull())
    .withColumn("current_date", f.current_date())
    .withColumn("difference", f.abs(f.datediff(f.col("DATE"), f.col("current_date"))))
)

window_spec = Window.partitionBy("PERSON_ID").orderBy("difference")

current_marker = (
    tmp.withColumn("row_number", f.row_number().over(window_spec))
    .filter(f.col("row_number") == 1).drop("difference", "row_number","current_date")
    .select("PERSON_ID",f.col("bmi").alias("BMI_DATE_CURRENT_MARKER"))
)

# Most recent BMI category to date of diagnosis
tmp = (
    bmi
    .withColumn("current_date", f.current_date())
    .withColumn("difference", f.abs(f.datediff(f.col("DATE"), f.col("current_date"))))
)

window_spec = Window.partitionBy("PERSON_ID").orderBy("difference")

current_category = (
    tmp.withColumn("row_number", f.row_number().over(window_spec))
    .filter(f.col("row_number") == 1).drop("difference", "row_number","current_date")
    .select("PERSON_ID",f.col("bmi_group").alias("BMI_DATE_CURRENT_CATEGORY"))
)

# COMMAND ----------

# MAGIC %md ##Combine

# COMMAND ----------

# Combine
bmi_final = (
    obese_bmi_no
    .join(obese_bmi_date_first,on="PERSON_ID",how="outer")
    .join(obese_bmi_date_last,on="PERSON_ID",how="outer")
    .join(obese_overweight_bmi_no,on="PERSON_ID",how="outer")
    .join(obese_overweight_bmi_date_first,on="PERSON_ID",how="outer")
    .join(obese_overweight_bmi_date_last,on="PERSON_ID",how="outer")
    .join(current_marker,on="PERSON_ID",how="outer")
    .join(current_category,on="PERSON_ID",how="outer")
    .join(diagnosis_marker,on="PERSON_ID",how="outer")
    .join(diagnosis_category,on="PERSON_ID",how="outer")
    .select("PERSON_ID",
            "BMI_OBESE_EVER","BMI_OBESE_NO","BMI_OBESE_DATE_FIRST","BMI_OBESE_DATE_LAST",
            "BMI_OBESE_OVERWEIGHT_EVER","BMI_OBESE_OVERWEIGHT_NO","BMI_OBESE_OVERWEIGHT_DATE_FIRST","BMI_OBESE_OVERWEIGHT_DATE_LAST",
            "BMI_DATE_CURRENT_MARKER","BMI_DATE_CURRENT_CATEGORY",
            "BMI_DATE_DIAGNOSIS_MARKER","BMI_DATE_DIAGNOSIS_CATEGORY"
            )
    .join(date_of_diagnosis,on="PERSON_ID",how="left")
    .withColumn("BMI_DATE_DIAGNOSIS_MARKER", f.when(f.col("date_of_diagnosis").isNull(),f.lit(None)).otherwise(f.col("BMI_DATE_DIAGNOSIS_MARKER")))
    .withColumn("BMI_DATE_DIAGNOSIS_CATEGORY", f.when(f.col("date_of_diagnosis").isNull(),f.lit(None)).otherwise(f.col("BMI_DATE_DIAGNOSIS_CATEGORY")))
    .drop("date_of_diagnosis")
)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=bmi_final, out_name=f'{proj}_data_preprocessing_bmi_{algorithm_timestamp}', save_previous=False)