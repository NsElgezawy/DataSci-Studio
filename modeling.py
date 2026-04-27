import findspark
findspark.init()

from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier,
    LinearSVC,
)
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    GeneralizedLinearRegression,
    IsotonicRegression,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)

# ── XGBoost ────────────────────────────────────────────────────────────────────
try:
    from xgboost.spark import SparkXGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    try:
        from sparkxgb import XGBoostClassifier as SparkXGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False


# ==============================================================================
# CLASSIFICATION
# ==============================================================================

def detect_task_type(df, target_col):
    """Returns 'binary' or 'multiclass'."""
    unique_count = df.select(target_col).distinct().count()
    return "binary" if unique_count == 2 else "multiclass"


def get_available_classification_models():
    models = [
        "Logistic Regression",
        "Random Forest",
        "GBT",
        "Decision Tree",
        "SVM (Linear)",
    ]
    if XGBOOST_AVAILABLE:
        models.append("XGBoost")
    return models


def train_logistic_regression(train_df, label_col, task_type="binary"):
    params = dict(labelCol=label_col, featuresCol="features", maxIter=100)
    if task_type == "multiclass":
        params["family"] = "multinomial"
    return LogisticRegression(**params).fit(train_df)


def train_random_forest_classifier(train_df, label_col, task_type="binary"):
    return RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        numTrees=100,
    ).fit(train_df)


def train_gbt_classifier(train_df, label_col, task_type="binary"):
    return GBTClassifier(
        labelCol=label_col,
        featuresCol="features",
        maxIter=50,
    ).fit(train_df)


def train_decision_tree_classifier(train_df, label_col, task_type="binary"):
    return DecisionTreeClassifier(
        labelCol=label_col,
        featuresCol="features",
    ).fit(train_df)


def train_svm(train_df, label_col, task_type="binary"):
    svc = LinearSVC(labelCol=label_col, featuresCol="features", maxIter=100)
    if task_type == "multiclass":
        from pyspark.ml.classification import OneVsRest
        ovr = OneVsRest(classifier=svc, labelCol=label_col, featuresCol="features")
        return ovr.fit(train_df)
    return svc.fit(train_df)


def train_xgboost(train_df, label_col, task_type="binary"):
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Run: pip install xgboost[spark]")
    num_class = train_df.select(label_col).distinct().count()
    params = dict(
        label_col=label_col,
        features_col="features",
        num_round=100,
        max_depth=6,
        eta=0.1,
    )
    if task_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["num_class"] = num_class
    else:
        params["objective"] = "binary:logistic"
    return SparkXGBClassifier(**params).fit(train_df)


def train_classification_model(model_name: str, train_df, label_col: str, task_type: str):
    dispatch = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest":       train_random_forest_classifier,
        "GBT":                 train_gbt_classifier,
        "Decision Tree":       train_decision_tree_classifier,
        "SVM (Linear)":        train_svm,
        "XGBoost":             train_xgboost,
    }
    if model_name not in dispatch:
        raise ValueError(f"Unknown classification model: {model_name}")
    return dispatch[model_name](train_df, label_col, task_type)


def evaluate_classification_model(model, test_df, label_col: str, task_type: str) -> dict:
    predictions = model.transform(test_df)
    mc_eval = MulticlassClassificationEvaluator(labelCol=label_col)

    def mc(metric):
        return mc_eval.evaluate(predictions, {mc_eval.metricName: metric})

    results = {
        "Accuracy":  mc("accuracy"),
        "F1":        mc("f1"),
        "Precision": mc("weightedPrecision"),
        "Recall":    mc("weightedRecall"),
    }

    if task_type == "binary":
        try:
            bin_eval = BinaryClassificationEvaluator(
                labelCol=label_col,
                rawPredictionCol="rawPrediction",
            )
            results["AUC-ROC"] = bin_eval.evaluate(
                predictions, {bin_eval.metricName: "areaUnderROC"}
            )
            results["AUC-PR"] = bin_eval.evaluate(
                predictions, {bin_eval.metricName: "areaUnderPR"}
            )
        except Exception:
            pass

    return results


def select_best_model(results: dict, primary_metric: str = "F1"):
    best = max(results, key=lambda n: results[n].get(primary_metric, 0))
    return best, results[best]


# ==============================================================================
# REGRESSION
# ==============================================================================

def get_available_regression_models():
    return [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "GBT Regressor",
        "Generalized Linear Regression",
        "Isotonic Regression",
    ]


def train_linear_regression(train_df, label_col):
    return LinearRegression(
        labelCol=label_col,
        featuresCol="features",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0,
    ).fit(train_df)


def train_decision_tree_regressor(train_df, label_col):
    return DecisionTreeRegressor(
        labelCol=label_col,
        featuresCol="features",
        maxDepth=5,
    ).fit(train_df)


def train_random_forest_regressor(train_df, label_col):
    return RandomForestRegressor(
        labelCol=label_col,
        featuresCol="features",
        numTrees=100,
        maxDepth=5,
    ).fit(train_df)


def train_gbt_regressor(train_df, label_col):
    return GBTRegressor(
        labelCol=label_col,
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
    ).fit(train_df)


def train_generalized_linear_regression(train_df, label_col):
    return GeneralizedLinearRegression(
        labelCol=label_col,
        featuresCol="features",
        family="gaussian",
        link="identity",
        maxIter=100,
    ).fit(train_df)


def train_isotonic_regression(train_df, label_col):
    # IsotonicRegression requires a single feature column
    return IsotonicRegression(
        labelCol=label_col,
        featuresCol="features",
        isotonic=True,
    ).fit(train_df)


def train_regression_model(model_name: str, train_df, label_col: str):
    dispatch = {
        "Linear Regression":              train_linear_regression,
        "Decision Tree Regressor":        train_decision_tree_regressor,
        "Random Forest Regressor":        train_random_forest_regressor,
        "GBT Regressor":                  train_gbt_regressor,
        "Generalized Linear Regression":  train_generalized_linear_regression,
        "Isotonic Regression":            train_isotonic_regression,
    }
    if model_name not in dispatch:
        raise ValueError(f"Unknown regression model: {model_name}")
    return dispatch[model_name](train_df, label_col)


def evaluate_regression_model(model, test_df, label_col: str) -> dict:
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")

    def reg(metric):
        return evaluator.evaluate(predictions, {evaluator.metricName: metric})

    results = {
        "RMSE":  reg("rmse"),
        "MAE":   reg("mae"),
        "R2":    reg("r2"),
        "MSE":   reg("mse"),
    }
    return results


def select_best_regression_model(results: dict, primary_metric: str = "R2"):
    # For regression, higher R2 is better, lower RMSE/MAE/MSE is better
    if primary_metric == "R2":
        best = max(results, key=lambda n: results[n].get(primary_metric, float('-inf')))
    else:
        best = min(results, key=lambda n: results[n].get(primary_metric, float('inf')))
    return best, results[best]