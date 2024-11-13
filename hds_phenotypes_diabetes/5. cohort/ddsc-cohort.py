# Databricks notebook source
# MAGIC %md
# MAGIC # Cohort
# MAGIC
# MAGIC **Description** This notebook curates the diabetes cohort in which a person is included if they have:
# MAGIC
# MAGIC - at least one diabetes code form primary or secondary care
# MAGIC - at least 6 months Insulin data
# MAGIC - have at least 2 high consecutive HbA1c markers
# MAGIC
# MAGIC All variables derived in the data_preprocessing stage are joined onto the cohort for use in the algorithm.
# MAGIC
# MAGIC Feature engineering is done here for use in the algorithm.
# MAGIC
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_cohort_{algorithm_timestamp}`**

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

# MAGIC %run "../0. parameters/ddsc-last_observable_date"

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

antidiabetic_df = spark.table(f'{dsa}.{proj}_data_preprocessing_medications_antidiabetic_{algorithm_timestamp}')
metformin_df = spark.table(f'{dsa}.{proj}_data_preprocessing_medications_metformin_{algorithm_timestamp}')
insulin_df = spark.table(f'{dsa}.{proj}_data_preprocessing_medications_insulin_{algorithm_timestamp}')

hba1c_df = spark.table(f'{dsa}.{proj}_data_preprocessing_hba1c_{algorithm_timestamp}')
bmi_df = spark.table(f'{dsa}.{proj}_data_preprocessing_bmi_{algorithm_timestamp}')

diabetes_df = spark.table(f'{dsa}.{proj}_data_preprocessing_diabetes_{algorithm_timestamp}')

date_of_diagnosis_df = spark.table(f'{dsa}.{proj}_data_preprocessing_date_of_diagnosis_{algorithm_timestamp}')

# COMMAND ----------

demographics = (
    spark.table(f'{dsa}.{proj}_curated_assets_demographics_{algorithm_timestamp}')
    .select("PERSON_ID",f.col("date_of_birth").alias("DATE_OF_BIRTH"),f.col("date_of_death").alias("DATE_OF_DEATH"),
                            f.col("sex").alias("SEX"),f.col("ethnicity_5_group").alias("ETHNICITY"),"in_gdppr")
)

# COMMAND ----------

insulin_6_months = insulin_df.filter(f.col("INSULIN_6_MONTHS")==1)

# COMMAND ----------

main_cohort = (

    diabetes_df
    .join(insulin_6_months,on="PERSON_ID",how="full")
    .join(hba1c_df,on="PERSON_ID",how="full")
    .select("PERSON_ID").distinct()

)

# COMMAND ----------

cohort = (

    main_cohort

    .join(diabetes_df,on="PERSON_ID",how="left")

    .join(metformin_df,on="PERSON_ID",how="left")
    .join(antidiabetic_df,on="PERSON_ID",how="left")
    .join(insulin_df,on="PERSON_ID",how="left")

    .join(hba1c_df,on="PERSON_ID",how="left")
    .join(bmi_df,on="PERSON_ID",how="left")
    .join(demographics,on="PERSON_ID",how="left")

    .join(date_of_diagnosis_df,on="PERSON_ID",how="left")

    

    .withColumn("TYPE1_EVER_PRIMARY",f.when(f.col("TYPE1_EVER_PRIMARY").isNull(),0).otherwise(f.col("TYPE1_EVER_PRIMARY")))
    .withColumn("TYPE2_EVER_PRIMARY",f.when(f.col("TYPE2_EVER_PRIMARY").isNull(),0).otherwise(f.col("TYPE2_EVER_PRIMARY")))
    .withColumn("NOS_EVER_PRIMARY",f.when(f.col("NOS_EVER_PRIMARY").isNull(),0).otherwise(f.col("NOS_EVER_PRIMARY")))
    .withColumn("OTHER_EVER_PRIMARY",f.when(f.col("OTHER_EVER_PRIMARY").isNull(),0).otherwise(f.col("OTHER_EVER_PRIMARY")))
    .withColumn("TYPE1_EVER_SECONDARY",f.when(f.col("TYPE1_EVER_SECONDARY").isNull(),0).otherwise(f.col("TYPE1_EVER_SECONDARY")))
    .withColumn("TYPE2_EVER_SECONDARY",f.when(f.col("TYPE2_EVER_SECONDARY").isNull(),0).otherwise(f.col("TYPE2_EVER_SECONDARY")))
    .withColumn("NOS_EVER_SECONDARY",f.when(f.col("NOS_EVER_SECONDARY").isNull(),0).otherwise(f.col("NOS_EVER_SECONDARY")))
    .withColumn("OTHER_EVER_SECONDARY",f.when(f.col("OTHER_EVER_SECONDARY").isNull(),0).otherwise(f.col("OTHER_EVER_SECONDARY")))

    .withColumn("TYPE1_NO_PRIMARY",f.when(f.col("TYPE1_NO_PRIMARY").isNull(),0).otherwise(f.col("TYPE1_NO_PRIMARY")))
    .withColumn("TYPE2_NO_PRIMARY",f.when(f.col("TYPE2_NO_PRIMARY").isNull(),0).otherwise(f.col("TYPE2_NO_PRIMARY")))
    .withColumn("NOS_NO_PRIMARY",f.when(f.col("NOS_NO_PRIMARY").isNull(),0).otherwise(f.col("NOS_NO_PRIMARY")))
    .withColumn("OTHER_NO_PRIMARY",f.when(f.col("OTHER_NO_PRIMARY").isNull(),0).otherwise(f.col("OTHER_NO_PRIMARY")))
    .withColumn("TYPE1_NO_SECONDARY",f.when(f.col("TYPE1_NO_SECONDARY").isNull(),0).otherwise(f.col("TYPE1_NO_SECONDARY")))
    .withColumn("TYPE2_NO_SECONDARY",f.when(f.col("TYPE2_NO_SECONDARY").isNull(),0).otherwise(f.col("TYPE2_NO_SECONDARY")))
    .withColumn("NOS_NO_SECONDARY",f.when(f.col("NOS_NO_SECONDARY").isNull(),0).otherwise(f.col("NOS_NO_SECONDARY")))
    .withColumn("OTHER_NO_SECONDARY",f.when(f.col("OTHER_NO_SECONDARY").isNull(),0).otherwise(f.col("OTHER_NO_SECONDARY")))

    .withColumn("TYPE1_EVER",f.when(f.col("TYPE1_EVER").isNull(),0).otherwise(f.col("TYPE1_EVER")))
    .withColumn("TYPE2_EVER",f.when(f.col("TYPE2_EVER").isNull(),0).otherwise(f.col("TYPE2_EVER")))
    .withColumn("NOS_EVER",f.when(f.col("NOS_EVER").isNull(),0).otherwise(f.col("NOS_EVER")))
    .withColumn("OTHER_EVER",f.when(f.col("OTHER_EVER").isNull(),0).otherwise(f.col("OTHER_EVER")))

    .withColumn("TYPE1_NO",f.when(f.col("TYPE1_NO").isNull(),0).otherwise(f.col("TYPE1_NO")))
    .withColumn("TYPE2_NO",f.when(f.col("TYPE2_NO").isNull(),0).otherwise(f.col("TYPE2_NO")))
    .withColumn("NOS_NO",f.when(f.col("NOS_NO").isNull(),0).otherwise(f.col("NOS_NO")))
    .withColumn("OTHER_NO",f.when(f.col("OTHER_NO").isNull(),0).otherwise(f.col("OTHER_NO")))

    .withColumn("flag_diabetes_cohort",f.when(f.col("flag_diabetes_cohort").isNull(),0).otherwise(f.col("flag_diabetes_cohort")))

    .withColumn("INSULIN_EVER",f.when(f.col("INSULIN_EVER").isNull(),0).otherwise(f.col("INSULIN_EVER")))
    .withColumn("ANTIDIABETIC_EVER",f.when(f.col("ANTIDIABETIC_EVER").isNull(),0).otherwise(f.col("ANTIDIABETIC_EVER")))
    .withColumn("METFORMIN_EVER",f.when(f.col("METFORMIN_EVER").isNull(),0).otherwise(f.col("METFORMIN_EVER")))

    .withColumn("INSULIN_NO",f.when(f.col("INSULIN_NO").isNull(),0).otherwise(f.col("INSULIN_NO")))
    .withColumn("ANTIDIABETIC_NO",f.when(f.col("ANTIDIABETIC_NO").isNull(),0).otherwise(f.col("ANTIDIABETIC_NO")))
    .withColumn("METFORMIN_NO",f.when(f.col("METFORMIN_NO").isNull(),0).otherwise(f.col("METFORMIN_NO")))

    .withColumn("INSULIN_6_MONTHS",f.when(f.col("INSULIN_6_MONTHS").isNull(),0).otherwise(f.col("INSULIN_6_MONTHS")))

    .withColumn("HBA1C_HIGH_EVER",f.when(f.col("HBA1C_HIGH_EVER").isNull(),0).otherwise(f.col("HBA1C_HIGH_EVER")))
    .withColumn("HBA1C_HIGH_NO",f.when(f.col("HBA1C_HIGH_NO").isNull(),0).otherwise(f.col("HBA1C_HIGH_NO")))

   .withColumn("flag_hba1c_cohort",f.when(f.col("flag_hba1c_cohort").isNull(),0).otherwise(f.col("flag_hba1c_cohort")))

    .withColumn("BMI_OBESE_EVER",f.when(f.col("BMI_OBESE_EVER").isNull(),0).otherwise(f.col("BMI_OBESE_EVER")))
    .withColumn("BMI_OBESE_NO",f.when(f.col("BMI_OBESE_NO").isNull(),0).otherwise(f.col("BMI_OBESE_NO")))

    .withColumn("BMI_OBESE_OVERWEIGHT_EVER",f.when(f.col("BMI_OBESE_OVERWEIGHT_EVER").isNull(),0).otherwise(f.col("BMI_OBESE_OVERWEIGHT_EVER")))
    .withColumn("BMI_OBESE_OVERWEIGHT_NO",f.when(f.col("BMI_OBESE_OVERWEIGHT_NO").isNull(),0).otherwise(f.col("BMI_OBESE_OVERWEIGHT_NO")))

    .withColumn("BMI_DATE_CURRENT_CATEGORY",f.when(f.col("BMI_DATE_CURRENT_CATEGORY").isNull(),f.lit("Unknown")).otherwise(f.col("BMI_DATE_CURRENT_CATEGORY")))

    .withColumn("BMI_DATE_DIAGNOSIS_CATEGORY",f.when(f.col("BMI_DATE_DIAGNOSIS_CATEGORY").isNull(),f.lit("Unknown")).otherwise(f.col("BMI_DATE_DIAGNOSIS_CATEGORY")))

    .withColumn("SEX",f.when(f.col("SEX").isNull(),f.lit("Unknown")).otherwise(f.col("SEX")))
    .withColumn("ETHNICITY",f.when(f.col("ETHNICITY").isNull(),f.lit("Unknown")).otherwise(f.col("ETHNICITY")))

    .withColumn("in_gdppr",f.when(f.col("in_gdppr").isNull(),0).otherwise(f.col("in_gdppr")))

    .filter(f.col('PERSON_ID').isNotNull())

)

# COMMAND ----------

# when PMEDS started
first_prescribing_date = (
    cohort.withColumn("first_prescribing_date", f.least(f.col("INSULIN_DATE_FIRST"), f.col("ANTIDIABETIC_DATE_FIRST"), f.col("METFORMIN_DATE_FIRST")))
    .select(f.min(f.col("first_prescribing_date")).alias("min_date")).collect()[0]["min_date"]
)

# COMMAND ----------

if individual_censor_dates_flag==True:
    individual_censor_dates = (spark.table(individual_censor_dates_table)
    )
else:
    individual_censor_dates = (demographics
                               .select("PERSON_ID",f.col("DATE_OF_DEATH").alias("CENSOR_END"))
                               .filter(f.col("CENSOR_END").isNotNull())
                               )

# COMMAND ----------

# Feature Engineering
cohort = (
    cohort

    .withColumn("first_prescribing_date",f.to_date(f.lit(first_prescribing_date), "yyyy-MM-dd"))
    .withColumn("prescribing_coverage_available", 
                f.when(f.col("DATE_OF_DEATH")<=f.col("first_prescribing_date"),0).otherwise(1))
    
    .join(individual_censor_dates, on="PERSON_ID", how="left")
    .withColumn("last_observable_date",f.to_date(f.lit(last_observable_date), "yyyy-MM-dd"))
    .withColumn("last_observable_date",f.when(f.col("CENSOR_END").isNull(),
                                              f.col("last_observable_date")).otherwise(f.col("CENSOR_END"))
    )

    .withColumn("last_insulin_to_last_observable_date_lt", 
                f. months_between(f.col("last_observable_date"),f.col("INSULIN_DATE_LAST")))
    .withColumn("currently_on_insulin",f.when(f.col("last_insulin_to_last_observable_date_lt")<=6,1).otherwise(0))
    .withColumn("not_currently_on_insulin",
                f.when(f.col("last_insulin_to_last_observable_date_lt").isNull(), 1)
                .when(f.col("last_insulin_to_last_observable_date_lt")>6, 1)
                .otherwise(0))
    
    .withColumn("date_diagnosis_to_last_observable_date_lt",
                f. months_between(f.col("last_observable_date"),f.col("date_of_diagnosis"))/12)
    .withColumn("more_than_3years_date_diagnosis_to_last_observable_date",
                f.when(f.col("date_diagnosis_to_last_observable_date_lt")>3,1).otherwise(0))
    
    .withColumn("age_at_diagnosis",
                (f.months_between(f.col("date_of_diagnosis"), f.col("DATE_OF_BIRTH")) / 12))
    
    .withColumn("diagnosis_to_insulin_lt", 
                f. months_between(f.col("INSULIN_DATE_FIRST"),f.col("date_of_diagnosis"))/12)
    .withColumn("on_insulin_within_1year",
                f.when(f.col("diagnosis_to_insulin_lt")<=1,1).otherwise(0))
    
    .withColumn("type1_more_recent_type2",
                f.when(f.col("TYPE1_NO") > f.col("TYPE2_NO"), 1).otherwise(0))
    
    .withColumn("type1_type2_ratio", f.col("TYPE1_NO") / f.col("TYPE2_NO")
                )
    
    .distinct()

)


# COMMAND ----------

# MAGIC %md #Save

# COMMAND ----------

save_table(df=cohort, out_name=f'{proj}_cohort_{algorithm_timestamp}', save_previous=False)