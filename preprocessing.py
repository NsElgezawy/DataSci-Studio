import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import io
import os
import pickle
import traceback
from datetime import datetime
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum, when, mean, trim, lower, to_date, regexp_replace, log1p, variance
from pyspark.sql.types import NumericType
import missingno as msno

# PySpark ML imports
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler, ChiSqSelector
from pyspark.sql.functions import min as spark_min
def count_duplicates_spark(df):

    total_rows = df.count()
    unique_rows = df.dropDuplicates().count()
    duplicates = total_rows - unique_rows
    return duplicates, total_rows, unique_rows

def remove_duplicates_spark(df):

    cleaned_df = df.dropDuplicates()
    return cleaned_df

def drop_rows_missing(df):

    return df.dropna()

def drop_rows_missing_cols(df, cols):

    return df.dropna(subset=cols)

def impute_mean(df, cols):

    for c in cols:
        mean_value = df.select(mean(c)).collect()[0][0]
        df = df.fillna({c: mean_value})
    return df

def impute_median(df, cols):

    for c in cols:
        median_value = df.approxQuantile(c, [0.5], 0.01)[0]
        df = df.fillna({c: median_value})
    return df

def impute_mode(df, cols):

    for c in cols:
        mode_value = df.groupBy(c).count().orderBy("count", ascending=False).first()[0]
        df = df.fillna({c: mode_value})
    return df

def forward_fill(df, target_col, order_col):

    w = Window.orderBy(order_col).rowsBetween(Window.unboundedPreceding, 0)
    df_filled = df.withColumn(target_col, F.last(target_col, ignorenulls=True).over(w))
    return df_filled

def backward_fill(df, target_col, order_col):

    w = Window.orderBy(order_col).rowsBetween(0, Window.unboundedFollowing)
    df_filled = df.withColumn(target_col, F.first(target_col, ignorenulls=True).over(w))
    return df_filled

def handle_inconsistent_typo(data, column, rare_threshold=5):

    data = data.withColumn(column, lower(trim(col(column))))
    unique_values = data.select(column).distinct()
    freq_table = data.groupBy(column).count().orderBy("count")
    suspicious = freq_table.filter(col("count") <= rare_threshold)
    return data, unique_values, freq_table, suspicious

def handle_invalid_min(data, column, min_val):

    invalid_rows = data.filter(col(column) <= min_val)
    return {
        "invalid_rows": invalid_rows,
        "invalid_count": invalid_rows.count(),
        'filtered_data_set': data.filter(col(column) > min_val)
    }

def check_unique_values(df, col_name):

    return df.select(col_name).distinct()

def clean_text_columns(df, cols):

    for c in cols:
        df = df.withColumn(c, lower(trim(col(c))))
    return df

def standardize_date(df, col_name, format="yyyy-MM-dd"):

    return df.withColumn(col_name, to_date(col(col_name), format))

def clean_numeric_text(df, col_name):

    df = df.withColumn(col_name, regexp_replace(col(col_name), "[^0-9.]", ""))
    return df

def split_data_spark(df, test_size=0.2, seed=42):

    train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)
    return train_df, test_df

def detect_columns(df, exclude_cols=[]):

    numeric_cols = [
        c for c, t in df.dtypes
        if t in ('int', 'double', 'float', 'long', 'bigint', 'decimal')
        and c not in exclude_cols
    ]
    categorical_cols = [
        c for c, t in df.dtypes
        if t == "string"
        and c not in exclude_cols
    ]
    return numeric_cols, categorical_cols

def apply_onehot_encoding_spark(train_df, test_df, col_name):

    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
    index_model = indexer.fit(train_df)
    train_indexed = index_model.transform(train_df)
    test_indexed = index_model.transform(test_df)
    encoder = OneHotEncoder(inputCols=[col_name + "_indexed"], outputCols=[col_name + "_ohe"])
    encoder_model = encoder.fit(train_indexed)
    train_encoded = encoder_model.transform(train_indexed)
    test_encoded = encoder_model.transform(test_indexed)
    return train_encoded, test_encoded

def apply_label_encoding_spark(train_df, test_df, col_name):

    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep")
    model = indexer.fit(train_df)
    train_encoded = model.transform(train_df)
    test_encoded = model.transform(test_df)
    return train_encoded, test_encoded

def frequency_encoding(train_df, test_df, cols):

    if len(cols) == 0:
        return train_df, test_df
    total = train_df.count()
    for c in cols:
        freq_df = train_df.groupBy(c).count()
        freq_df = freq_df.withColumn(c + "_freq", F.col("count") / total).select(c, c + "_freq")
        train_df = train_df.join(freq_df, on=c, how="left")
        test_df = test_df.join(freq_df, on=c, how="left")
        train_df = train_df.withColumnRenamed(c + "_freq", c).drop(c)
        test_df = test_df.withColumnRenamed(c + "_freq", c).drop(c)
    return train_df, test_df

def apply_standard_scaler_spark(train_df, test_df):

    numeric_cols = [col_name for col_name, col_type in train_df.dtypes if col_type in ('int', 'double', 'float', 'long', 'bigint', 'decimal')]
    if not numeric_cols:
        return train_df, test_df
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)
    scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(train_df)
    train_scaled = scaler_model.transform(train_df)
    test_scaled = scaler_model.transform(test_df)
    return train_scaled, test_scaled

def apply_minmax_scaler_spark(train_df, test_df):

    numeric_cols = [col_name for col_name, col_type in train_df.dtypes if col_type in ('int', 'double', 'float', 'long', 'bigint', 'decimal')]
    if not numeric_cols:
        return train_df, test_df
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)
    scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_features")
    scaler_model = scaler.fit(train_df)
    train_scaled = scaler_model.transform(train_df)
    test_scaled = scaler_model.transform(test_df)
    return train_scaled, test_scaled

def select_features_variance_spark(train_df, threshold=0.01):

    numeric_cols = [c for c, t in train_df.dtypes if t in ('int', 'double', 'float', 'long', 'bigint', 'decimal')]
    variances = train_df.select([variance(col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()
    selected_features = [col_name for col_name, var_value in variances.items() if var_value is not None and var_value > threshold]
    return selected_features

def select_features_chisq_spark(train_df, label_col, k=7):

    numeric_cols = [c for c, t in train_df.dtypes if t in ('int', 'double', 'float', 'long', 'bigint', 'decimal')]
    feature_cols = [c for c in numeric_cols if c != label_col and train_df.select(spark_min(col(c))).collect()[0][0] >= 0]
    if len(feature_cols) < 2:
        return feature_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
    train_vec = assembler.transform(train_df)
    selector = ChiSqSelector(numTopFeatures=min(k, len(feature_cols)), featuresCol="features_vec", outputCol="selected_features", labelCol=label_col)
    model = selector.fit(train_vec)
    selected_indices = model.selectedFeatures
    selected_features = [feature_cols[i] for i in selected_indices if i < len(feature_cols)]
    return selected_features