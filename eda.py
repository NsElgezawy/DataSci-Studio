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
def get_shape(data):

    return (data.count(), len(data.columns))

def get_n_of_rows(data):

    return data.count()

def get_n_of_columns(data):

    return len(data.columns)

def get_columns(data):

    return data.columns

def get_rows(data, n=10):

    return data.show(n)

def get_describe(data):

    return data.describe().show()

def get_schema(data):

    return data.printSchema()

# ─── Visualization Functions ──────────────────────────────────────────────────
def histogram_plot(data, column, sample_fraction=0.05):

    df = data.select(column).dropna().sample(False, sample_fraction).toPandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], bins=40, kde=False, color='#7c6af7', ax=ax)
    ax.set_title(f"Histogram of {column}", color='#e8e8f0')
    ax.set_xlabel(column, color='#888899')
    ax.set_ylabel("Frequency", color='#888899')
    return fig

def kde_plot(data, column, sample_fraction=0.05):

    df = data.select(column).dropna().sample(False, sample_fraction).toPandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(df[column], fill=True, color='#6af7c2', ax=ax)
    ax.set_title(f"KDE Plot of {column}", color='#e8e8f0')
    ax.set_xlabel(column, color='#888899')
    ax.set_ylabel("Density", color='#888899')
    return fig

def box_plot(data, column, sample_fraction=0.05):

    df = data.select(column).dropna().sample(False, sample_fraction).toPandas()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df[column], color='#f7c26a', ax=ax)
    ax.set_title(f"Box Plot of {column}", color='#e8e8f0')
    ax.set_xlabel(column, color='#888899')
    return fig

def countplot(data, column, top_n=20):

    pdf = (data.groupBy(column)
           .count()
           .orderBy("count", ascending=False)
           .limit(top_n)
           .toPandas())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pdf[column].astype(str), pdf['count'], color='#7c6af7')
    ax.set_xticklabels(pdf[column].astype(str), rotation=45, ha='right')
    ax.set_title(f"Countplot - {column}", color='#e8e8f0')
    ax.set_xlabel(column, color='#888899')
    ax.set_ylabel("Count", color='#888899')
    ax.tick_params(colors='#888899')
    return fig

def to_pandas_safe(df, sample_size=0.1):

    return df.sample(fraction=sample_size).toPandas()

def plot_heatmap(df, sample_size=0.2):

    df_pd = to_pandas_safe(df, sample_size)
    corr = df_pd.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax,
                fmt='.2f', annot_kws={'color': '#e8e8f0'})
    ax.set_title("Correlation Heatmap", color='#e8e8f0')
    ax.tick_params(colors='#888899')
    return fig

def plot_scatter(df, x_col, y_col, sample_size=0.1):

    df_pd = to_pandas_safe(df, sample_size)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df_pd, color='#7c6af7', alpha=0.6, ax=ax)
    ax.set_title(f"{x_col} vs {y_col}", color='#e8e8f0')
    ax.set_xlabel(x_col, color='#888899')
    ax.set_ylabel(y_col, color='#888899')
    ax.tick_params(colors='#888899')
    return fig

def plot_pair(df, sample_size=0.1):

    df_pd = to_pandas_safe(df, sample_size)
    df_pd = df_pd.select_dtypes(include="number")


    if len(df_pd.columns) > 5:
        df_pd = df_pd.iloc[:, :5]

    g = sns.pairplot(df_pd, diag_kind='kde', plot_kws={'alpha': 0.5, 'color': '#7c6af7'})
    return g.fig

def plot_missing(df, sample_size=0.3):

    df_pd = to_pandas_safe(df, sample_size)
    fig, ax = plt.subplots(figsize=(10, 5))
    msno.matrix(df_pd, ax=ax, color=(0.486, 0.416, 0.969), sparkline=False)
    ax.set_title("Missing Values Matrix", color='#e8e8f0')
    ax.tick_params(colors='#888899')
    return fig

def plot_violin(df, x_col, y_col, sample_size=0.1):

    df_pd = to_pandas_safe(df, sample_size)
    fig, ax = plt.subplots(figsize=(10, 6))


    top_categories = (df_pd[x_col].value_counts().head(20).index)
    df_filtered = df_pd[df_pd[x_col].isin(top_categories)]

    sns.violinplot(x=x_col, y=y_col, data=df_filtered, palette='viridis', ax=ax)
    ax.set_title(f"{y_col} distribution by {x_col}", color='#e8e8f0')
    ax.set_xlabel(x_col, color='#888899')
    ax.set_ylabel(y_col, color='#888899')
    ax.tick_params(colors='#888899', axis='x', rotation=45)
    return fig

def plot_stacked_bar(df, index_col, cols, stacked=True, sample_size=0.2):

    df_pd = to_pandas_safe(df, sample_size)


    top_categories = (df_pd[index_col].value_counts().head(20).index)
    df_filtered = df_pd[df_pd[index_col].isin(top_categories)]

    df_plot = df_filtered.groupby(index_col)[cols].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='bar', stacked=stacked, ax=ax, colormap='viridis')
    ax.set_title("Stacked Bar Chart" if stacked else "Grouped Bar Chart", color='#e8e8f0')
    ax.set_ylabel("Values", color='#888899')
    ax.set_xlabel(index_col, color='#888899')
    ax.tick_params(colors='#888899', axis='x', rotation=45)
    ax.legend(loc='best', facecolor='#111118', labelcolor='#e8e8f0')
    return fig

# ─── Missing Detection ────────────────────────────────────────────────────────
def check_missing_values(df, extra_missing=None):

    if extra_missing is None:
        extra_missing = ["na", "null", "n/a"]

    df_cached = df.cache()
    total_rows = df_cached.count()

    missing_exprs = []
    for c in df.columns:
        cond = (
            col(c).isNull() |
            (trim(col(c)) == "") |
            (lower(trim(col(c))).isin(extra_missing))
        )
        missing_exprs.append(sum(when(cond, 1).otherwise(0)).alias(c))

    missing_df = df_cached.select(missing_exprs)

    missing_df = missing_df.selectExpr(
        "stack({0}, {1}) as (column_name, missing_count)".format(
            len(df.columns),
            ", ".join([f"'{c}', {c}" for c in df.columns])
        )
    )

    missing_df = missing_df.withColumn(
        "missing_percentage",
        (col("missing_count") / total_rows) * 100
    )

    return missing_df.orderBy(col("missing_count").desc())

# ─── Duplicates Detection ─────────────────────────────────────────────────────
def check_duplicates(df, subset=None):

    df_cached = df.cache()
    total_rows = df_cached.count()

    if subset:
        grouped = df_cached.groupBy(subset).count()
    else:
        grouped = df_cached.groupBy(df.columns).count()

    duplicates_df = grouped.filter(col("count") > 1)
    duplicate_rows = duplicates_df.selectExpr("sum(count - 1) as dup").collect()[0]["dup"]

    if duplicate_rows is None:
        duplicate_rows = 0

    duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0

    result = df.sparkSession.createDataFrame([
        (total_rows, duplicate_rows, duplicate_percentage)
    ], ["total_rows", "duplicate_rows", "duplicate_percentage"])

    return result

def get_duplicate_groups(df, subset=None):

    if subset is None:
        subset = df.columns

    duplicates = (df.groupBy(subset)
                  .count()
                  .filter(col("count") > 1))
    return duplicates

# ─── Outlier Detection ────────────────────────────────────────────────────────
def iqr_outlier_detection(data, column):

    q1, q3 = data.approxQuantile(column, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data.filter(
        (data[column] < lower_bound) | (data[column] > upper_bound))
    outliers_number = outliers.count()

    return outliers, outliers_number, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):

    stats = data.select(
        F.mean(column).alias("mean"),
        F.stddev(column).alias("std")
    ).collect()[0]

    mean_val = stats["mean"]
    std_val = stats["std"]

    outliers = data.filter(
        F.abs((F.col(column) - mean_val) / std_val) > threshold
    )
    outliers_number = outliers.count()

    return outliers, outliers_number, mean_val, std_val

def detect_outliers_percentile(data, column, lower_percentile=0.01, upper_percentile=0.99):

    bounds = data.approxQuantile(column, [lower_percentile, upper_percentile], 0.01)
    lower = bounds[0]
    upper = bounds[1]

    outliers = data.filter((data[column] < lower) | (data[column] > upper))
    outliers_number = outliers.count()

    return outliers, outliers_number, lower, upper