# Databricks notebook source
# MAGIC %sql
# MAGIC DROP CATALOG otto_unity CASCADE;

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup Tasks

# COMMAND ----------

# MAGIC %pip install "mlflow>=2.11.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
import pyspark.pandas as ps

# COMMAND ----------

# DBTITLE 1,Create catalog
# MAGIC %sql
# MAGIC -- Create a catalog to start
# MAGIC CREATE CATALOG otto_unity;

# COMMAND ----------

# DBTITLE 1,Set default catalog and schema
catalog_name = "otto_unity"
schema_name = "default"
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# DBTITLE 1,Create a Volume to hold data
# MAGIC %sql
# MAGIC --Volumes are a container for unstructured data
# MAGIC CREATE VOLUME IF NOT EXISTS default.myVolume;

# COMMAND ----------

# DBTITLE 1,Load data into Volume
raw_data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv", sep=";")
raw_data.rename(columns = {'fixed acidity':'fixed_acidity', 'volatile acidity':'volatile_acidity', 'citric acid':'citric_acid', 'residual sugar':'residual_sugar', 'free sulfur dioxide':'free_sulfer_dioxide', 'total sulfur dioxide':'total_sulfer_dioxide'}, inplace = True)
df = spark.createDataFrame(raw_data)
volume_path = f"/Volumes/{catalog_name}/default/myVolume/myData"
df.write.mode("overwrite").save(volume_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data from Postgres and a Volume into a Table 

# COMMAND ----------

displayHTML('<img width="400px" src ="/files/tables/lineage_image.png">')

# COMMAND ----------

# DBTITLE 1,Loading data from volume to a table
feature_table = spark.read.load(volume_path)
feature_table.write.mode("overwrite").saveAsTable("otto_unity.default.feature_table", overwrite=True)

# COMMAND ----------

# DBTITLE 1,Adding data from Postgres into table
# MAGIC %sql
# MAGIC -- add data from Postgres into feature table
# MAGIC INSERT INTO default.feature_table (quality)
# MAGIC     SELECT id FROM postgres.market_data.example_table

# COMMAND ----------

# MAGIC %md
# MAGIC # Train an MLflow model from feature table

# COMMAND ----------

displayHTML('<img width="400px" src ="/files/tables/lineage_image2.png">')

# COMMAND ----------

# DBTITLE 1,Register MLflow model
spark = SparkSession.builder.getOrCreate()
spark_df = spark.table("default.feature_table")
data = spark_df.toPandas()
mlflow.set_registry_uri("databricks-uc")
registered_model_name = f"{catalog_name}.{schema_name}.test_quality"
print(registered_model_name)

# COMMAND ----------

# DBTITLE 1,Import libraries
import functools
import logging
from typing import Any, Dict, Optional

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException

from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.databricks_managed_catalog_service_pb2 import UnityCatalogService
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_in_databricks_runtime
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

# COMMAND ----------

# DBTITLE 1,Split data into training and test sets
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = 0.5
l1_ratio = 0.5

# COMMAND ----------

# DBTITLE 1,Train model
with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    dataset2 = mlflow.data.load_delta(table_name="feature_table", version="0")

    mlflow.log_input(dataset2, context="training")
    # Infer model signature, because ML models in UC
    signature = mlflow.models.infer_signature(model_input=test_x[:10], model_output=predicted_qualities[:10])
    
    # Passing registered_model_name here triggers registration of the ML model
    # as a new model version in UC, under the registered model with the specified three-level name
    mlflow.sklearn.log_model(lr, "model", registered_model_name=registered_model_name, signature=signature)

# COMMAND ----------

# DBTITLE 1,Train a second model version
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        dataset2 = mlflow.data.load_delta(table_name="murt_test.default.winequality", version="0")

        mlflow.log_input(dataset2, context="training")
        # Infer model signature, because ML models in UC
        signature = mlflow.models.infer_signature(model_input=test_x[:10], model_output=predicted_qualities[:10])
        
        # Passing registered_model_name here triggers registration of the ML model
        # as a new model version in UC, under the registered model with the specified three-level name
        mlflow.sklearn.log_model(lr, "model", registered_model_name=registered_model_name, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bring-your-own lineage

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add custom first-mile lineage metadata

# COMMAND ----------

displayHTML('<img width="400px" src ="/files/tables/lineage_image6.png">')

# COMMAND ----------

import requests, json

def make_request(tpe, data):
  url = "https://e2-demo-field-eng.cloud.databricks.com/api/2.0/lineage-tracking/custom"
  headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {dbutils.entry_point.getDbutils().notebook().getContext().apiToken().get()}'
  }
  res = requests.request(tpe, url, headers=headers, data=json.dumps(data)).text
  print(json.dumps(json.loads(res), indent=2))

def byol_create(data): make_request("POST", data)
def byol_update(data): make_request("PATCH", data)
def byol_delete(data): make_request("DELETE", data)
def byol_list(data): make_request("GET", data)

# COMMAND ----------

def table(name): return f'{catalog_name}.{schema_name}.{name}'

# COMMAND ----------

byol_create({
  "entities": [
    {
      "entity_id": {
        "provider_type": "CUSTOM",
        "guid": "A1234"
      },
      "entity_type": "Salesforce",
      "display_name": "CRM Accounts",
      "url": "https://salesforce.com/",
      "description": "Accounts Object from SFDC",
      "properties": """{"owner": "john.doe@company.com", "account_type": "install_partner"}"""
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "CUSTOM",
        "guid": "A1234"
      },
      "target": {
        "provider_type": "DATABRICKS",
        "databricks_type": "PATH",
        "guid": "dbfs:/Volumes/otto_unity/default/myVolume/myData"
      }
    }
  ]
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add custom last-mile lineage metadata

# COMMAND ----------

byol_create({
  "entities": [
    {
      "entity_id": {
        "provider_type": "CUSTOM",
        "guid": "A123456"
      },
      "entity_type": "Tableau",
      "display_name": "Insights Dashboard",
      "url": "https://tableau.com/",
      "description": "Insights dashboard in Tableau",
      "properties": """{}"""
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "DATABRICKS",
        "databricks_type": "TABLE",
        "guid": "otto_unity.default.feature_table"
      },
      "target": {
        "provider_type": "CUSTOM",
        "guid": "A123456"
      }
    }
  ]
})

# COMMAND ----------

volume_path = f"/Volumes/otto_unity/default/myVolume/myData";
byol_delete({
  "entities": [
    {
      "provider_type": "CUSTOM",
      "guid": "A1234"
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "CUSTOM",
        "guid": "A1234"
      },
      "target": {
        "provider_type": "DATABRICKS",
        "databricks_type": "PATH",
        "guid": volume_path
      }
    }
  ]
})

# COMMAND ----------

catalog_name = "otto_unity"
schema_name = "default"
byol_delete({
  "entities": [
    {
      "provider_type": "CUSTOM",
      "guid": "A123456"
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "DATABRICKS",
        "databricks_type": "TABLE",
        "guid": table("feature_table")
      },
      "target": {
        "provider_type": "CUSTOM",
        "guid": "1"
      }
    }
  ]
})

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE FUNCTION density(region STRING)
# MAGIC RETURN IF(IS_ACCOUNT_GROUP_MEMBER('admin'), true, region="US");
# MAGIC

# COMMAND ----------

byol_create({
  "entities": [
    {
      "entity_id": {
        "provider_type": "CUSTOM",
        "guid": "A123"
      },
      "entity_type": "Salesforce",
      "display_name": "CRM Accounts",
      "url": "https://salesforce.com/",
      "description": "Accounts Object from SFDC",
      "properties": """{"owner": "john.doe@company.com", "account_type": "install_partner"}"""
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "CUSTOM",
        "guid": "A123"
      },
      "target": {
        "provider_type": "DATABRICKS",
        "databricks_type": "PATH",
        "guid": "otto.uc.customers"
      }
    }
  ]
})

# COMMAND ----------

byol_create({
  "entities": [
    {
      "entity_id": {
        "provider_type": "CUSTOM",
        "guid": "B1234"
      },
      "entity_type": "Tableau",
      "display_name": "Insights Dashboard",
      "url": "https://tableau.com/",
      "description": "Insights dashboard in Tableau",
      "properties": """{}"""
    }
  ],
  "relationships": [
    {
      "source": {
        "provider_type": "DATABRICKS",
        "databricks_type": "TABLE",
        "guid": "otto.uc.customers"
      },
      "target": {
        "provider_type": "CUSTOM",
        "guid": "B1234"
      }
    }
  ]
})

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE FUNCTION ssn_mask(ssn STRING)
# MAGIC   RETURN CASE WHEN is_member('HumanResourceDept') THEN ssn ELSE '***-**-****' END;
