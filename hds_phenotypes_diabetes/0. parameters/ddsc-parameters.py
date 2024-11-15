# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # DDSC Parameters for Defining Diabetes Algorithm
# MAGIC
# MAGIC **How to use** Set your own project paramters in this notebook to run the diabetes algorithm.
# MAGIC Go to 3.1 Set Project Specific Varaibles and set your:
# MAGIC - project name and date for which tables will be saved
# MAGIC - individual censor end dates if appropriate
# MAGIC
# MAGIC **Parent Project** DDSC
# MAGIC
# MAGIC **Description** This notebook defines a set of parameters, which are loaded in each notebook in the data curation pipeline, so that helper functions and parameters are consistently available.
# MAGIC
# MAGIC **Author(s)** Tom Bolton, Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Acknowledgements** Based on CCU003_05-D01-parameters and CCU002_07
# MAGIC
# MAGIC **Notes** This pipeline has an initial production date, set at `pipeline_production_date`, and the `archived_on` dates used for each dataset correspond to the latest (most recent) batch of data before this date. Should the pipeline and all the notebooks that follow need to be updated and rerun, then this notebook should be rerun with an updated `pipeline_production_date`. Note that if a parameters_df_datasets table already exists using your project and pipeline_production_date then the Datasets section of this notebook will be skipped.
# MAGIC
# MAGIC **DDSC Versions** 
# MAGIC <br>Version 1 as at '2024-03-17'
# MAGIC <br>Version 2 as at '2024-07-25'
# MAGIC
# MAGIC Versions for end of year reclassification
# MAGIC - 2022-01-01 (for 2021 end of year)
# MAGIC - 2023-01-01 (for 2022 end of year)
# MAGIC - 2024-01-01 (for 2023 end of year)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`proj_parameters_df_datasets`**: table of `archived_on` dates for each dataset that can be used consistently throughout the pipeline 

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Setup

# COMMAND ----------

spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

db = 'dars_nic_391419_j3w9t'
dbc = f'{db}_collab'
dsa = f'dsa_391419_j3w9t_collab'

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Libraries

# COMMAND ----------

import pyspark.sql.functions as f
import pandas as pd
import re
import datetime
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Common Functions

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Set Project Specific Variables

# COMMAND ----------

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
proj = 'ddsc_new' #e.g. ccu001_01 (you might also want to set this as a unique identifier specific to your study end date too eg ccu000_2023)


# -----------------------------------------------------------------------------
# Pipeline production date
# -----------------------------------------------------------------------------
# date at which pipeline was created and archived_on dates for datasets have been selected based on
# Note that the earliest archived_on date for the data above is 2020-12-11; your pipeline_production_date must not be before this date
pipeline_production_date = '2024-07-25'

# Do note that a last_observable_date is generated to account for data lag in the run up to pipeline_production_date

# -----------------------------------------------------------------------------
# Study End Date
# -----------------------------------------------------------------------------
# censor end date if individual_censor_dates not supplied. This can be the same as your pipeline_production_date but it can also be earlier
# For example if you wanted to find the diabetes status as at 2020-01-01 you must set your study end date to this but pick a 
# pipeline_production_date that is later than 2020-12-11
study_end_date = '2024-07-25'

# note that if last_observable_date < study_end_date then then study_end_date will be set to last_observable_date

# -----------------------------------------------------------------------------
# Individual censor end dates (dates at which a persons diabetes type will be classified)
# -----------------------------------------------------------------------------

# if you have individual censor end dates set this to True, else False
individual_censor_dates_flag = False

# if individual_censor_dates_flag = True, set the db and table name for the censor dates e.g. fc_test_diabetes_censor_end. Your table should contain 2 columns, PERSON_ID and CENSOR_END. If individual_censor_dates_flag = False then set individual_censor_dates_table = None
individual_censor_dates_table = f'dsa_391419_j3w9t_collab.fc_test_diabetes_censor_end'

# COMMAND ----------


if (datetime.datetime.strptime(pipeline_production_date, "%Y-%m-%d")) >= (datetime.datetime.strptime("2020-12-11", "%Y-%m-%d")):
  print("pipeline_production_date valid.")
else:
  raise ValueError("pipeline_production_date should be later than 2020-12-11.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Algorithm Version

# COMMAND ----------

date_obj = datetime.datetime.strptime(pipeline_production_date, '%Y-%m-%d')

# Extract the year and month
year = date_obj.year
month = f'{date_obj.month:02}'
day = f'{date_obj.day:02}'

algorithm_timestamp = f'{year}_{month}_{day}'
print(algorithm_timestamp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Parameters Table Name

# COMMAND ----------

parameters_df_name = f'{proj}_ddsc_parameters_df_datasets_{algorithm_timestamp}'

# COMMAND ----------

try:
    # Check if table exists already
    df = spark.table(f'{dsa}.{parameters_df_name}')
    does_not_exist_toggle = False
except AnalysisException as e:
    does_not_exist_toggle = True

# COMMAND ----------

# If table exists already then the Datasets section of this notebook will be skipped
print(does_not_exist_toggle)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Dataset Paths

# COMMAND ----------

# -----------------------------------------------------------------------------
# Dates
# -----------------------------------------------------------------------------
study_start_date = '1900-01-01' 
study_end_date   = pipeline_production_date


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
# data frame of datasets
datasets = [ 
    ['gdppr',         dbc, f'gdppr_{db}_archive',             'NHS_NUMBER_DEID',                'DATE']  
  , ['hes_apc',       dbc, f'hes_apc_all_years_archive',      'PERSON_ID_DEID',                 'EPISTART']
  , ['deaths',        dbc, f'deaths_{db}_archive',            'DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'REG_DATE_OF_DEATH']
  , ['pmeds',         dbc, f'primary_care_meds_{db}_archive', 'Person_ID_DEID',                 'ProcessingPeriodDate']            
]

tmp_df_datasets = pd.DataFrame(datasets, columns=['dataset', 'database', 'table', 'id', 'date']).reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Datasets Archived States

# COMMAND ----------

# for each dataset: 
#   find the latest (most recent) archived_on date before the pipeline_production_date
#   create a table containing a row with the latest archived_on date and count of the number of records for each dataset
  
if(does_not_exist_toggle):

  latest_archived_on = []
  for index, row in tmp_df_datasets.iterrows():
    # initial  
    dataset = row['dataset']
    path = row['database'] + '.' + row['table']
    print(index, dataset, path); print()

    # point to table
    tmpd = spark.table(path).select("archived_on")
    
    tmpa = (
      tmpd
      .distinct()
      .where(f.col('archived_on') <= pipeline_production_date)
      .orderBy(f.desc('archived_on'))
      .limit(1)
      .withColumn('dataset', f.lit(dataset))
      .select('dataset', 'archived_on')      
      .toPandas()      
    )
    tmpaa = tmpa.iloc[0]['archived_on']

    # get count
    tmpn = tmpd.where(f.col('archived_on') == tmpaa).count()
    tmpa['n'] = tmpn
    print(tmpa.to_string())

    # append results
    if(index == 0): latest_archived_on = tmpa
    else: latest_archived_on = pd.concat([latest_archived_on, tmpa])


# COMMAND ----------

if(does_not_exist_toggle):
  
  tmp_df_datasets_sp = spark.createDataFrame(tmp_df_datasets) 
  latest_archived_on_sp = spark.createDataFrame(latest_archived_on) 
  parameters_df_datasets = tmp_df_datasets_sp.join(
                                 latest_archived_on_sp,on="dataset",how="left"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save

# COMMAND ----------

if(does_not_exist_toggle):
    save_table(df=parameters_df_datasets, out_name=parameters_df_name, save_previous=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import

# COMMAND ----------

spark.sql(f'REFRESH TABLE {dsa}.{parameters_df_name}')
parameters_df_datasets = (
  spark.table(f'{dsa}.{parameters_df_name}')
  .withColumn("archived_on", f.col("archived_on").cast("string"))
  .withColumn("archived_on", f.to_date(f.col("archived_on"), "yyyy-MM-dd"))
  .withColumn("n", f.format_number(f.col("n"), 0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Demographics Table

# COMMAND ----------

batch_name = (
    sqlContext
        .sql("show tables in dsa_391419_j3w9t_collab")
        .select("tableName")
        .filter("tableName LIKE '%hds_curated_assets__demographics%'")
        .filter(~f.col('tableName').contains('pre'))
        .orderBy(f.col("tableName").desc())
        .withColumn("date_str", f.regexp_extract("tableName", r"(\d{4}_\d{2}_\d{2})", 1))
        .withColumn("date", f.to_date("date_str", "yyyy_MM_dd"))
        .filter(f.col("date")<=pipeline_production_date)
        .limit(1)
        .collect()[0][0]
        )

path_demographics = f'{dsa}.{batch_name}'
print(f'Demographics Table Version: ',path_demographics)