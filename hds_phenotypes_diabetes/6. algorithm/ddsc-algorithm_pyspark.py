# Databricks notebook source
# MAGIC %md
# MAGIC # Algorithm - PySpark
# MAGIC
# MAGIC **Description** This notebook runs the DDSC Algorithm in PySpark
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_cohort_out_{algorithm_timestamp}`**

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

# MAGIC %md # Data

# COMMAND ----------

# Cohort In
cohort = spark.table(f'{dsa}.{proj}_cohort_{algorithm_timestamp}')

# COMMAND ----------

cohort_in_dd_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/data_dictionaries/ddsc-algorithm-cohort_in_data_dictionary.csv'

paths = [cohort_in_dd_path]
spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

cohort_in_dd_path = codelist

display(cohort_in_dd_path)

# COMMAND ----------

# MAGIC %md # Algorithm

# COMMAND ----------

cohort = (
    cohort
    # Create ANTIDIABETIC_METFORMIN_EVER column
    .withColumn(
        "ANTIDIABETIC_METFORMIN_EVER", 
        f.when((f.col("ANTIDIABETIC_EVER") == 1) | (f.col("METFORMIN_EVER") == 1), 1).otherwise(0)
    )
    # Create at_least_two column
    .withColumn(
        "at_least_two", 
        f.when(((f.col("INSULIN_EVER") + f.col("HBA1C_HIGH_EVER") + f.col("ANTIDIABETIC_METFORMIN_EVER")) >= 2),1).otherwise(0)
    )
)

# COMMAND ----------

# Step 1: Any Diabetes Other codes
cohort = cohort.withColumn("step_1", f.when(f.col("OTHER_EVER") == 1, "Yes").otherwise("No"))

# Step 2: Prescribing data coverage available
cohort = cohort.withColumn("step_2", 
    f.when((f.col("step_1") == "No") & (f.col("prescribing_coverage_available") == 1), "Yes")
    .when((f.col("step_1") == "No") & (f.col("prescribing_coverage_available") != 1), "No")
    .otherwise(None))

# Step 3: Not currently on Insulin AND >3 years from diagnosis to last observable date
cohort = cohort.withColumn("step_3", 
    f.when((f.col("step_2") == "Yes") & (f.col("not_currently_on_insulin") == 1) & (f.col("more_than_3years_date_diagnosis_to_last_observable_date") == 1), "Yes")
    .when((f.col("step_2") == "Yes") & ~((f.col("not_currently_on_insulin") == 1) & (f.col("more_than_3years_date_diagnosis_to_last_observable_date") == 1)), "No")
    .otherwise(None))

# Step 3.1: Any Diabetes Type 2 codes
cohort = cohort.withColumn("step_3_1", 
    f.when((f.col("step_3") == "Yes") & (f.col("TYPE2_EVER") == 1), "Yes")
    .when((f.col("step_3") == "Yes") & (f.col("TYPE2_EVER") != 1), "No")
    .otherwise(None))

# Step 4: Diabetes Type 1 codes and NO Diabetes Type 2 codes
cohort = cohort.withColumn("step_4", 
    f.when(((f.col("step_2") == "No") | (f.col("step_3") == "No")) & (f.col("TYPE1_EVER") == 1) & (f.col("TYPE2_EVER") == 0), "Yes")
    .when(((f.col("step_2") == "No") | (f.col("step_3") == "No")) & ~((f.col("TYPE1_EVER") == 1) & (f.col("TYPE2_EVER") == 0)), "No")
    .otherwise(None))

# Step 5: Diabetes Type 2 codes and NO Diabetes Type 1 codes
cohort = cohort.withColumn("step_5", 
    f.when((f.col("step_4") == "No") & (f.col("TYPE1_EVER") == 0) & (f.col("TYPE2_EVER") == 1), "Yes")
    .when((f.col("step_4") == "No") & ~((f.col("TYPE1_EVER") == 0) & (f.col("TYPE2_EVER") == 1)), "No")
    .otherwise(None))

# Step 6: Diabetes Type 1 codes and Diabetes Type 2 codes
cohort = cohort.withColumn("step_6", 
    f.when((f.col("step_5") == "No") & (f.col("TYPE1_EVER") == 1) & (f.col("TYPE2_EVER") == 1), "Yes")
    .when((f.col("step_5") == "No") & ~((f.col("TYPE1_EVER") == 1) & (f.col("TYPE2_EVER") == 1)), "No")
    .otherwise(None))

# Step 6.1: Diabetes Type 1 code more recent than Diabetes Type 2 code
cohort = cohort.withColumn("step_6_1", 
    f.when((f.col("step_6") == "Yes") & (f.col("type1_more_recent_type2") == 1), "Yes")
    .when((f.col("step_6") == "Yes") & (f.col("type1_more_recent_type2") != 1), "No")
    .otherwise(None))

# Step 7: Diagnosed age <35 years and on Insulin within 1 year of diagnosis
cohort = cohort.withColumn("step_7", 
    f.when((f.col("step_6") == "No") & (f.col("age_at_diagnosis") < 35) & (f.col("on_insulin_within_1year") == 1), "Yes")
    .when((f.col("step_6") == "No") & ~((f.col("age_at_diagnosis") < 35) & (f.col("on_insulin_within_1year") == 1)), "No")
    .otherwise(None))

# Step 8: Any Diabetes NOS codes
cohort = cohort.withColumn("step_8", 
    f.when(((f.col("step_7") == "No") | (f.col("step_3_1") == "No")) & (f.col("NOS_EVER") == 1), "Yes")
    .when(((f.col("step_7") == "No") | (f.col("step_3_1") == "No")) & (f.col("NOS_EVER") != 1), "No")
    .otherwise(None))

# Step 9: 2 of the following: Insulin, high HbA1c, Antidiabetic/Metformin
cohort = cohort.withColumn("step_9", 
    f.when((f.col("step_8") == "No") & (f.col("at_least_two")==1), "Yes")
    .when((f.col("step_8") == "No") & (f.col("at_least_two")!=1), "No")
    .otherwise(None))

# COMMAND ----------

# MAGIC %md # Results

# COMMAND ----------

# Create Diabetes Variable
cohort = cohort.withColumn(
    "out_diabetes",
    f.when(f.col("step_1") == "Yes", "Other")
    .when(
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "Yes") & (f.col("step_6_1") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "No") & (f.col("step_6") == "Yes") & (f.col("step_6_1") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "No") & (f.col("step_7") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "No") & (f.col("step_7") == "Yes")
        ),
        "Type 1"
    )
    .when(
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "Yes") & 
            (f.col("step_3_1") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "Yes")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "No") & (f.col("step_6") == "Yes") & (f.col("step_6_1") == "No")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "Yes") & (f.col("step_6_1") == "No")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "Yes") & (f.col("step_6_1") == "No")
        ),
        "Type 2"
    )
    
    .when(
    
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "No") & 
            (f.col("step_7") == "No") & (f.col("step_8") == "Yes") 
        ) | 
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "No") & 
            (f.col("step_7") == "No") & (f.col("step_8") == "No") & (f.col("step_9") == "Yes") #
        ) | 
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "Yes") & 
            (f.col("step_3_1") == "No") & (f.col("step_8") == "Yes")
        ) | 
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "Yes") & 
            (f.col("step_3_1") == "No") & (f.col("step_8") == "No") & (f.col("step_9") == "Yes")
        ) | 
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "No") & (f.col("step_6") == "No") & (f.col("step_7") == "No") & 
            (f.col("step_8") == "Yes")
        ) | 
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "No") & (f.col("step_6") == "No") & (f.col("step_7") == "No") & 
            (f.col("step_8") == "No") & (f.col("step_9") == "Yes")
        ),
        "NOS"
    )
    .when(
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "No") & 
            (f.col("step_4") == "No") & (f.col("step_5") == "No") & (f.col("step_6") == "No") & 
            (f.col("step_7") == "No") & (f.col("step_8") == "No") & (f.col("step_9") == "No") #
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "Yes") & (f.col("step_3") == "Yes") & 
            (f.col("step_3_1") == "No") & (f.col("step_8") == "No") & (f.col("step_9") == "No")
        ) |
        (
            (f.col("step_1") == "No") & (f.col("step_2") == "No") & (f.col("step_4") == "No") & 
            (f.col("step_5") == "No") & (f.col("step_6") == "No") & (f.col("step_7") == "No") & 
            (f.col("step_8") == "No") & (f.col("step_9") == "No")
        ),
        "Unlikely"
    )
    .otherwise(None)
)


# COMMAND ----------

results = cohort.groupBy("out_diabetes").count()
display(results)

# COMMAND ----------

results_with_sdc = (
    results
    .withColumn("n_pct", f.round(f.col("count") / f.sum("count").over(Window.partitionBy()) * 100, 2))
    .withColumn("tmp", f.when(f.col("out_diabetes") == "Unlikely", None).otherwise(f.col("count")))
    .withColumn("n_pct_of_diabetes_cohort", f.round(f.col("tmp") / f.sum("tmp").over(Window.partitionBy()) * 100, 2))
    .drop("tmp")
    .withColumn("count", f.when(f.col("count") < 10, 10).otherwise(f.round(f.col("count") / 5) * 5))
    .withColumnRenamed("count", "n")
)

display(results_with_sdc)

# COMMAND ----------

# MAGIC %md # Save

# COMMAND ----------

cohort = (
    cohort
    .select("PERSON_ID","out_diabetes","date_of_diagnosis","age_at_diagnosis",
            "step_1","step_2","step_3","step_3_1","step_4","step_5","step_6","step_6_1","step_7","step_8","step_9")
    )

# COMMAND ----------

save_table(df=cohort, out_name=f'{proj}_cohort_out_{algorithm_timestamp}', save_previous=True)

# COMMAND ----------

cohort_out_dd_path = f'/Workspace{os.path.dirname(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))}/data_dictionaries/ddsc-algorithm-cohort_out_data_dictionary.csv'

paths = [cohort_out_dd_path]
spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

cohort_out_dd_path = codelist

display(cohort_out_dd_path)