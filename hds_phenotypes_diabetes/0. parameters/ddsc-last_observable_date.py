# Databricks notebook source
# MAGIC %md
# MAGIC # Last Observable Date
# MAGIC
# MAGIC **Description** This notebook derives the last observable date for those who are alive as at the project pipeline end date. That is, it works out the date at which GDPPR, HES APC and PMEDs collectively have no coverage lag.
# MAGIC
# MAGIC For each dataset, the lag is quantified by computing a rolling average of the data counts over a 3-month period, which considers the current month and the two preceding months. Periods of coverage lag are identified by comparing the actual count for each month/year to this rolling average. If the actual count falls below 80% of the rolling average, that month/year is flagged as experiencing a coverage lag.
# MAGIC
# MAGIC From the most recent month backwards, the date at which coverage was last considered full is the first month in which the count is not less than the lag theshbold (set at 0.8 here) * the rolling average.
# MAGIC
# MAGIC Note that in addition to checking date coverage, diagnostic coding coverage is checked for HES APC these can sometimes take an extra month to come through after the dates.
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

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

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

parameters_df_last_observable_date_name = f'{proj}_parameters_df_last_observable_date_{algorithm_timestamp}'

try:
    # Check if table exists already
    df = spark.table(f'{dsa}.{parameters_df_last_observable_date_name}')
    does_not_exist_toggle = False
except AnalysisException as e:
    does_not_exist_toggle = True

# If table exists already then the Datasets section of this notebook will be skipped
print(does_not_exist_toggle)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

if(does_not_exist_toggle):
    gdppr = extract_batch_from_archive(parameters_df_datasets, 'gdppr')
    hes_apc = extract_batch_from_archive(parameters_df_datasets, 'hes_apc')
    pmeds = extract_batch_from_archive(parameters_df_datasets, 'pmeds')

    max_archived_on_date = parameters_df_datasets.select("archived_on").agg(f.max(f.col("archived_on")).alias("max_date")).collect()[0]["max_date"]

    three_years_ago = f.date_sub(f.lit(max_archived_on_date), 3 * 365)

    # Filter for dates in the last three years
    gdppr_df = gdppr.filter(f.col("DATE") >= three_years_ago).filter(f.col("DATE")<=max_archived_on_date)
    hes_apc_df = hes_apc.filter(f.col("EPISTART") >= three_years_ago).filter(f.col("EPISTART")<=max_archived_on_date)
    pmeds_df = pmeds.withColumnRenamed('ProcessingPeriodDate', 'DATE').filter(f.col("DATE") >= three_years_ago).filter(f.col("DATE")<=max_archived_on_date)

# COMMAND ----------

if(does_not_exist_toggle):
    from pyspark.sql.types import DateType
    min_date = three_years_ago
    max_date = f.add_months(f.lit(max_archived_on_date), 1)

    # Generate the sequence of dates - used for months where no coverage recently so can impute with 0
    date_seq_df = spark.range(1).select(
        f.explode(
        f.sequence(
            f.lit(min_date).cast(DateType()),
            f.lit(max_date).cast(DateType()),
            f.expr("interval 1 month")
        )
        ).alias("date")
        ).withColumn("date", f.date_trunc("month", f.col("date"))).withColumn("date", f.date_format(f.col("date"), "yyyy-MM-dd"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## GDPPR

# COMMAND ----------

if(does_not_exist_toggle):
        window_spec = Window.orderBy("DATE").rowsBetween(-2, 0)
        lagging_threshold = 0.8

        
        gdppr_coverage = (gdppr_df
                          .withColumn("DATE", f.date_trunc("month", f.col("DATE")))
                          .withColumn("DATE", f.date_format(f.col("DATE"), "yyyy-MM-dd"))
                          .groupBy("DATE").count()
                          .withColumn("rolling_avg", f.avg("count").over(window_spec))
                          .withColumn("is_lagging", f.col("count") < (lagging_threshold * f.col("rolling_avg"))
                        )
        )

        gdppr_coverage = date_seq_df.withColumnRenamed("date","DATE").join(gdppr_coverage,on="DATE",how="left")

        display(gdppr_coverage)

# COMMAND ----------

if(does_not_exist_toggle):
        gdppr_last_observable_date = (
            gdppr_coverage.filter(f.col("is_lagging")=="false").orderBy(f.col("DATE").desc())
            .limit(1).select("DATE")
            # add on 1 month so can filter on <= below this
            .withColumn("DATE", f.add_months(f.col("DATE"), 1))
            .collect()[0][0]
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ## HES APC

# COMMAND ----------

if(does_not_exist_toggle):
        window_spec = Window.orderBy("EPISTART").rowsBetween(-2, 0)
        lagging_threshold = 0.8

        
        hes_apc_coverage = (hes_apc_df
                          .withColumn("EPISTART", f.date_trunc("month", f.col("EPISTART")))
                          .withColumn("EPISTART", f.date_format(f.col("EPISTART"), "yyyy-MM-dd"))
                          .groupBy("EPISTART").count()
                          .withColumn("rolling_avg", f.avg("count").over(window_spec))
                          .withColumn("is_lagging", f.col("count") < (lagging_threshold * f.col("rolling_avg"))
                        )
        )

        hes_apc_coverage = date_seq_df.withColumnRenamed("date","EPISTART").join(hes_apc_coverage,on="EPISTART",how="left")

        display(hes_apc_coverage)


# COMMAND ----------

if(does_not_exist_toggle):
        hes_apc_last_observable_date = (
            hes_apc_coverage.filter(f.col("is_lagging")=="false").orderBy(f.col("EPISTART").desc())
            .limit(1).select("EPISTART")
            # add on 1 month so can filter on <= below this
            .withColumn("EPISTART", f.add_months(f.col("EPISTART"), 1))
            .collect()[0][0]
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnostic codes
# MAGIC
# MAGIC

# COMMAND ----------

if(does_not_exist_toggle):
    
    # Prepare long format of HES APC
    _hes_apc = (
        hes_apc_df  
        .select(['PERSON_ID_DEID', 'EPIKEY', 'EPISTART', 'ADMIDATE'] 
          + [col for col in list(hes_apc.columns) if re.match(r'^DIAG_(3|4)_\d\d$', col)])
        .withColumnRenamed('PERSON_ID_DEID', 'PERSON_ID')
        .orderBy('PERSON_ID', 'EPIKEY')
        )

    hes_apc_long = (
        reshape_wide_to_long_multi(_hes_apc, i=['PERSON_ID', 'EPIKEY', 'EPISTART', 'ADMIDATE'], j='POSITION', stubnames=['DIAG_4_', 'DIAG_3_'])
        .withColumn('_tmp', f.substring(f.col('DIAG_4_'), 1, 3))
        .withColumn('_chk', udf_null_safe_equality('DIAG_3_', '_tmp').cast(t.IntegerType()))
        .withColumn('_DIAG_4_len', f.length(f.col('DIAG_4_')))
        .withColumn('_chk2', f.when((f.col('_DIAG_4_len').isNull()) | (f.col('_DIAG_4_len') <= 4), 1).otherwise(0))
        )
    
    hes_apc_long = (
        hes_apc_long.drop('_tmp', '_chk'))

    hes_apc_long = reshape_wide_to_long_multi(hes_apc_long, i=['PERSON_ID', 'EPIKEY', 'EPISTART', 'ADMIDATE', 'POSITION'], j='DIAG_DIGITS', stubnames=['DIAG_'])\
        .withColumnRenamed('POSITION', 'DIAG_POSITION')\
        .withColumn('DIAG_POSITION', f.regexp_replace('DIAG_POSITION', r'^[0]', ''))\
        .withColumn('DIAG_DIGITS', f.regexp_replace('DIAG_DIGITS', r'[_]', ''))\
        .withColumn('DIAG_', f.regexp_replace('DIAG_', r'X$', ''))\
        .withColumn('DIAG_', f.regexp_replace('DIAG_', r'[.,\-\s]', ''))\
        .withColumnRenamed('DIAG_', 'CODE')\
        .where((f.col('CODE').isNotNull()) & (f.col('CODE') != ''))\
        .orderBy(['PERSON_ID', 'EPIKEY', 'DIAG_DIGITS', 'DIAG_POSITION'])



    window_spec = Window.orderBy("EPISTART").rowsBetween(-2, 0)
    lagging_threshold = 0.8

        
    hes_apc_coverage_diagnostic = (hes_apc_long
                          .withColumn("EPISTART", f.date_trunc("month", f.col("EPISTART")))
                          .withColumn("EPISTART", f.date_format(f.col("EPISTART"), "yyyy-MM-dd"))
                          .groupBy("EPISTART").count()
                          .withColumn("rolling_avg", f.avg("count").over(window_spec))
                          .withColumn("is_lagging", f.col("count") < (lagging_threshold * f.col("rolling_avg"))
                        )
        )

    hes_apc_coverage_diagnostic = date_seq_df.withColumnRenamed("date","EPISTART").join(hes_apc_coverage_diagnostic,on="EPISTART",how="left")

    display(hes_apc_coverage_diagnostic)

# COMMAND ----------

if(does_not_exist_toggle):
        hes_apc_diagnostic_last_observable_date = (
            hes_apc_coverage_diagnostic.filter(f.col("is_lagging")=="false").orderBy(f.col("EPISTART").desc())
            .limit(1).select("EPISTART")
            # add on 1 month so can filter on <= below this
            .withColumn("EPISTART", f.add_months(f.col("EPISTART"), 1))
            .collect()[0][0]
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ##PMEDS

# COMMAND ----------

if(does_not_exist_toggle):
        window_spec = Window.orderBy("DATE").rowsBetween(-2, 0)
        lagging_threshold = 0.8

        
        pmeds_coverage = (pmeds_df
                          .withColumn("DATE", f.date_trunc("month", f.col("DATE")))
                          .withColumn("DATE", f.date_format(f.col("DATE"), "yyyy-MM-dd"))
                          .groupBy("DATE").count()
                          .withColumn("rolling_avg", f.avg("count").over(window_spec))
                          .withColumn("is_lagging", f.col("count") < (lagging_threshold * f.col("rolling_avg"))
                        )
        )

        pmeds_coverage = date_seq_df.withColumnRenamed("date","DATE").join(pmeds_coverage,on="DATE",how="left")

        display(pmeds_coverage)

# COMMAND ----------

if(does_not_exist_toggle):
        pmeds_last_observable_date = (
            pmeds_coverage.filter(f.col("is_lagging")=="false").orderBy(f.col("DATE").desc())
            .limit(1).select("DATE")
            # add on 1 month so can filter on <= below this
            .withColumn("DATE", f.add_months(f.col("DATE"), 1))
            .collect()[0][0]
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall

# COMMAND ----------

if(does_not_exist_toggle):
    earliest_date = min(gdppr_last_observable_date,hes_apc_last_observable_date,hes_apc_diagnostic_last_observable_date)
    earliest_date_str = earliest_date.strftime("%Y-%m-%d")

    overall_last_observable_date = earliest_date

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

if(does_not_exist_toggle):
    data = [("gdppr", gdppr_last_observable_date),
         ("hes_apc", hes_apc_last_observable_date),
         ("hes_apc_diagnostic", hes_apc_diagnostic_last_observable_date),
         ("pmeds", pmeds_last_observable_date),
         ("all", overall_last_observable_date)]
    columns = ["dataset", "last_observable_date"]
    parameters_df_last_observable_date = spark.createDataFrame(data, columns)

# COMMAND ----------

if(does_not_exist_toggle):
    save_table(df=parameters_df_last_observable_date, 
               out_name=parameters_df_last_observable_date_name, save_previous=True)

# COMMAND ----------

parameters_df_last_observable_date = spark.table(f'{dsa}.{parameters_df_last_observable_date_name}')
last_observable_date = parameters_df_last_observable_date.filter(f.col("dataset")=="all").select("last_observable_date").collect()[0][0]

# COMMAND ----------

last_observable_date