# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing - HbA1c cohort
# MAGIC
# MAGIC **Description** This notebook curates the HbA1c cohort in which someone must have at least 2 consecutive high HbA1c readings. The first 2 consecutive high HbA1c readings must have been within 2 years of each other.
# MAGIC
# MAGIC The following data cleaning/pre-processing has been done:
# MAGIC - readings before DOB removed
# MAGIC - records that are not high removed (i.e. readings < 48mmol/mol)
# MAGIC - remove those who do not have at least 2 consecutive high HbA1c readings in their history 
# MAGIC
# MAGIC Feature enginerring is then performed to establish the following:
# MAGIC - If at least 2 consecutive high readings were ever found in a person's history
# MAGIC - The number of high readings found in a person's histroy, after at least 2 consecutive readings have been found (e.g. if a high reading was found in Jan 2018 and Jan 2021 then a low reading was found in Mar 2021 then a high found in June 2021, Aug 2021, Feb 2022 then the number of high readings recorded would be 3. There was more than 2 years between Jan 2018 and Jan 2021 and in between Jan 2021 and June 2021 there was a low reading)
# MAGIC - The date at which the first high reading was found in a person's histroy (in the above example this would be June 2021)
# MAGIC - The date at which the last high reading was found in a person's history
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`ddsc_data_preprocessing_hba1c_long_{algorithm_timestamp}`**
# MAGIC - **`ddsc_data_preprocessing_hba1c_{algorithm_timestamp}`**

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

hba1c = spark.table(f'{dsa}.{proj}_curated_assets_hba1c_{algorithm_timestamp}')

# COMMAND ----------

demographics = spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')

# COMMAND ----------

# MAGIC %md #Curate

# COMMAND ----------

hba1c_cleaned = (
        hba1c
        .join((demographics.select(f.col("PERSON_ID"),"date_of_birth")),how="left",on="PERSON_ID")
        .filter(f.col("DATE")>=f.col("date_of_birth"))
        .drop("date_of_birth")
)

# COMMAND ----------

window_spec = Window.partitionBy("PERSON_ID").orderBy("DATE")

tmp = hba1c_cleaned
tmp = (tmp
       .withColumn("prev_hba1c_group", f.lag("hba1c_group").over(window_spec))
       .withColumn("consecutive_high", 
                   f.when((f.col("hba1c_group") == "High") & (f.col("prev_hba1c_group") == "High"), 1).otherwise(0))
)

# ensure the first 2 consecutive dates are within 2 years
window_spec_date = Window.partitionBy("PERSON_ID").orderBy("DATE")

tmp = (
    tmp
    .withColumn("PREV_DATE", f.lag("DATE").over(window_spec_date))
    .withColumn("DATE_DIFF", f.when(f.col("PREV_DATE").isNotNull(), f.datediff(f.col("DATE"), f.col("PREV_DATE")) / 365.25).otherwise(None))
)

# COMMAND ----------

window_spec_row = Window.partitionBy("PERSON_ID").orderBy("DATE")

tmp = tmp.withColumn("row_num", f.row_number().over(window_spec_row))

first_consecutive_high = (
    tmp.filter( (f.col("consecutive_high") == 1) & (f.col("DATE_DIFF")<=2) )
                           .groupBy("PERSON_ID")
                           .agg({"DATE": "min"})
                           .withColumnRenamed("min(DATE)", "first_consecutive_high_date")
                           )

# COMMAND ----------

tmp = (
    tmp
    .join(first_consecutive_high, "PERSON_ID", "left")
    .withColumn("first_consecutive_high_flag_second", 
                   f.when((f.col("DATE") == f.col("first_consecutive_high_date")) & (f.col("consecutive_high") == 1), 1).otherwise(0))
    .drop("prev_hba1c_group", "consecutive_high", "row_num", "first_consecutive_high_date")
)

hba1c_high_with_flag = (
    (hba1c_cleaned.filter(f.col("hba1c_group")=="High"))
    .join( (tmp.drop("hba1c_group").filter(f.col('first_consecutive_high_flag_second')==1)),on=["PERSON_ID","DATE","hba1c"],how="left")
    .orderBy(["PERSON_ID","DATE"])
    )

# COMMAND ----------

# remove people in the high cohort who do have high records but never 2 consecutive
hba1c_high_consec_only = (
    hba1c_high_with_flag
    .filter(f.col('first_consecutive_high_flag_second')==1).select("PERSON_ID")
    .join(hba1c_high_with_flag,on="PERSON_ID",how="left")
    .orderBy(["PERSON_ID","DATE"])
    )

window_spec = Window.partitionBy("PERSON_ID").orderBy("DATE")

hba1c_high_consec_only_first = (
    hba1c_high_consec_only.withColumn("NEXT_FLAG", f.lag("first_consecutive_high_flag_second", -1).over(window_spec))
    .withColumn("first_consecutive_high_flag", (f.col("NEXT_FLAG") == 1).cast("int"))
    .drop("NEXT_FLAG","first_consecutive_high_flag_second")
)


window_spec = Window.partitionBy("PERSON_ID").orderBy("DATE").rowsBetween(Window.unboundedPreceding, Window.currentRow)

hba1c_high_consec_only_first=(hba1c_high_consec_only_first.withColumn("has_high_flag", f.max(f.when(f.col("first_consecutive_high_flag") == 1, 1).otherwise(0)).over(window_spec)))

# Filter the rows to keep only those after the first occurrence of 1
hba1c_high_consec_only_first = (
    hba1c_high_consec_only_first.filter(f.col("has_high_flag") == 1)
    .drop("has_high_flag","PREV_DATE","DATE_DIFF")
)

# COMMAND ----------

display(hba1c_high_consec_only_first.orderBy(["PERSON_ID","DATE"]))

# COMMAND ----------

# Date of first record
high_hba1c_date_first = (
    hba1c_high_consec_only_first
    .filter(f.col("first_consecutive_high_flag")==1)
    .select("PERSON_ID",f.col("DATE").alias("HBA1C_HIGH_DATE_FIRST"))
)


# No of High Records in history
high_hba1c_no = (
    hba1c_high_consec_only_first.select("PERSON_ID","DATE").groupBy("PERSON_ID").count().withColumnRenamed("count","HBA1C_HIGH_NO")
    .withColumn("HBA1C_HIGH_EVER",f.lit(1))
)

# Last Date
_win_last = Window\
  .partitionBy('PERSON_ID')\
  .orderBy(f.col('DATE').desc())

high_hba1c_date_last = (
    hba1c_high_consec_only_first.select("PERSON_ID","DATE")
    .withColumn('_rownum', f.row_number().over(_win_last))
    .where(f.col('_rownum') == 1)
    .drop("_rownum")
    .withColumnRenamed("DATE","HBA1C_HIGH_DATE_LAST")
)

# Combine
hba1c_high = (
    high_hba1c_no
    .join(high_hba1c_date_first,on="PERSON_ID",how="outer")
    .join(high_hba1c_date_last,on="PERSON_ID",how="outer")
    .select("PERSON_ID","HBA1C_HIGH_EVER","HBA1C_HIGH_NO","HBA1C_HIGH_DATE_FIRST","HBA1C_HIGH_DATE_LAST")
    # flag for if in hba1cs cohort
    .withColumn("flag_hba1c_cohort",f.lit(1))
)

# COMMAND ----------

display(hba1c_high)

# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=hba1c_high_consec_only_first, out_name=f'{proj}_data_preprocessing_hba1c_long_{algorithm_timestamp}', save_previous=False)

# COMMAND ----------

save_table(df=hba1c_high, out_name=f'{proj}_data_preprocessing_hba1c_{algorithm_timestamp}', save_previous=False)