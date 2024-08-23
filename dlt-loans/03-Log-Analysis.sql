-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC # DLT pipeline log analysis
-- MAGIC
-- MAGIC <img style="float:right" width="500" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/retail-dlt-data-quality-dashboard.png">
-- MAGIC
-- MAGIC Each DLT Pipeline saves events and expectations metrics in the Storage Location defined on the pipeline. From this table we can see what is happening and the quality of the data passing through it.
-- MAGIC
-- MAGIC You can leverage the expecations directly as a SQL table with Databricks SQL to track your expectation metrics and send alerts as required. 
-- MAGIC
-- MAGIC This notebook extracts and analyses expectation metrics to build such KPIS.
-- MAGIC
-- MAGIC ## Accessing the Delta Live Table pipeline events with Unity Catalog
-- MAGIC
-- MAGIC Databricks provides an `event_log` function which is automatically going to lookup the event log table. You can specify any table to get access to the logs:
-- MAGIC
-- MAGIC `SELECT * FROM event_log(TABLE(catalog.schema.my_table))`
-- MAGIC
-- MAGIC #### Using Legacy hive_metastore
-- MAGIC *Note: If you are not using Unity Catalog (legacy hive_metastore), you can find your event log location opening the Settings of your DLT pipeline, under `storage` :*
-- MAGIC
-- MAGIC ```
-- MAGIC {
-- MAGIC     ...
-- MAGIC     "name": "lakehouse_churn_dlt",
-- MAGIC     "storage": "/demos/dlt/loans",
-- MAGIC     "target": "your schema"
-- MAGIC }
-- MAGIC ```
-- MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
-- MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-engineering&org_id=1444828305810485&notebook=%2F03-Log-Analysis&demo_name=dlt-loans&event=VIEW&path=%2F_dbdemos%2Fdata-engineering%2Fdlt-loans%2F03-Log-Analysis&version=1">

-- COMMAND ----------

SELECT * FROM event_log(TABLE(otto.dlt-full.raw_txs))

-- COMMAND ----------

CREATE OR REPLACE TEMPORARY VIEW demo_dlt_loans_system_event_log_raw 
  as SELECT * FROM event_log(TABLE(otto.dlt-full.raw_txs));
SELECT * FROM demo_dlt_loans_system_event_log_raw order by timestamp desc;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The `details` column contains metadata about each Event sent to the Event Log. There are different fields depending on what type of Event it is. Some examples include:
-- MAGIC * `user_action` Events occur when taking actions like creating the pipeline
-- MAGIC * `flow_definition` Events occur when a pipeline is deployed or updated and have lineage, schema, and execution plan information
-- MAGIC   * `output_dataset` and `input_datasets` - output table/view and its upstream table(s)/view(s)
-- MAGIC   * `flow_type` - whether this is a complete or append flow
-- MAGIC   * `explain_text` - the Spark explain plan
-- MAGIC * `flow_progress` Events occur when a data flow starts running or finishes processing a batch of data
-- MAGIC   * `metrics` - currently contains `num_output_rows`
-- MAGIC   * `data_quality` - contains an array of the results of the data quality rules for this particular dataset
-- MAGIC     * `dropped_records`
-- MAGIC     * `expectations`
-- MAGIC       * `name`, `dataset`, `passed_records`, `failed_records`
-- MAGIC   

-- COMMAND ----------

-- DBTITLE 1,Lineage Information
SELECT
  details:flow_definition.output_dataset,
  details:flow_definition.input_datasets,
  details:flow_definition.flow_type,
  details:flow_definition.schema,
  details:flow_definition
FROM demo_dlt_loans_system_event_log_raw
WHERE details:flow_definition IS NOT NULL
ORDER BY timestamp

-- COMMAND ----------

-- DBTITLE 1,Data Quality Results
SELECT
  id,
  expectations.dataset,
  expectations.name,
  expectations.failed_records,
  expectations.passed_records
FROM(
  SELECT 
    id,
    timestamp,
    details:flow_progress.metrics,
    details:flow_progress.data_quality.dropped_records,
    explode(from_json(details:flow_progress:data_quality:expectations
             ,schema_of_json("[{'name':'str', 'dataset':'str', 'passed_records':42, 'failed_records':42}]"))) expectations
  FROM demo_dlt_loans_system_event_log_raw
  WHERE details:flow_progress.metrics IS NOT NULL) data_quality

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Your expectations are ready to be queried in SQL! Open the <a dbdemos-dashboard-id="dlt-expectations" href='/sql/dashboardsv3/01ef5ab1b1bc1019b73afff409c190a7' target="_blank">data Quality Dashboard example</a> for more details.
