import os
import streamlit as st

import findspark
findspark.init()
from pyspark.sql import SparkSession


@st.cache_resource
def get_spark_session():
    spark = SparkSession.builder \
        .appName("DataSciStudio") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    return spark


spark = get_spark_session()


def load_csv_spark(file, nrows=None):
    df = spark.read.csv(file, header=True, inferSchema=True)
    if nrows and nrows > 0:
        df = df.limit(nrows)
    return df


def load_kaggle_spark(path: str, filename: str, nrows=None):
    full_path = os.path.join(path.strip(), filename.strip())
    if not os.path.exists(full_path):
        dataset_id = path.strip().rstrip('/')
        os.system(f"kaggle datasets download -d {dataset_id} --unzip -p /tmp/kaggle_data/")
        full_path = f"/tmp/kaggle_data/{filename.strip()}"
    df = spark.read.csv(full_path, header=True, inferSchema=True)
    if nrows and nrows > 0:
        df = df.limit(nrows)
    return df