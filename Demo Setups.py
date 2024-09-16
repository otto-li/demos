# Databricks notebook source
# MAGIC %pip install dbdemos
# MAGIC

# COMMAND ----------

import dbdemos
dbdemos.list_demos()

# COMMAND ----------

import dbdemos
dbdemos.install('lakehouse-iot-platform', catalog='otto', schema='iot', overwrite=True)

# COMMAND ----------

import dbdemos
dbdemos.install('uc-01-acl', catalog='otto', schema='uc', overwrite=True)

# COMMAND ----------

import dbdemos
dbdemos.install('dlt-loans', catalog='otto', schema='dlt-full', overwrite=True)

# COMMAND ----------

import dbdemos
dbdemos.install('mlops-end2end', catalog='otto', schema='mlops', overwrite = True)
