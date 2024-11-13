# Databricks notebook source
# MAGIC %md
# MAGIC # Curated Assets - Demographics
# MAGIC
# MAGIC **Description** This notebook curates a table detailing an individual's most recent demographic data
# MAGIC
# MAGIC The HDS curated asset is used.
# MAGIC
# MAGIC **Author(s)** Fionna Chalmers (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`ddsc_curated_assets_demographics_{algorithm_timestamp}`** 

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

# COMMAND ----------

# MAGIC %run "../../shds/common/functions"

# COMMAND ----------

# MAGIC %run "../0. parameters/ddsc-parameters"

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

demographics = spark.table(path_demographics)
display(demographics)

# COMMAND ----------

# MAGIC %md # Save

# COMMAND ----------

save_table(df=demographics, out_name=f'{proj}_curated_assets_demographics_{algorithm_timestamp}', save_previous=False)