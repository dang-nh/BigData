"""
Train & evaluate ER classifier on labeled pairs.
Output: same_as_edges.parquet — entity pairs determined to be the same.
"""

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

PARQUET_DIR = os.getenv("PARQUET_OUTPUT_DIR", "/data/parquet")
FEATURES_DIR = f"{PARQUET_DIR}/er_features"
GOLD_DIR = f"{PARQUET_DIR}/er_gold"
SAME_AS_OUT = f"{PARQUET_DIR}/same_as_edges"

MED_FEATURE_COLS = ["jw_drug", "lev_drug", "token_overlap", "jaccard_sim", "same_ndc_product"]
DIAG_FEATURE_COLS = ["jw_title", "lev_title", "token_overlap", "jaccard_sim", "same_icd_category"]


def build_gold_labels(spark: SparkSession, entity_type: str) -> DataFrame:
    """
    Build gold set using heuristics:
    - Positive: same_ndc_product=1 AND jw_drug > 0.9 (for meds)
    - Negative: same blocking key but jw < 0.5
    Fall back to this if no manual gold file exists.
    """
    feat_df = spark.read.parquet(f"{FEATURES_DIR}/{entity_type}")

    if entity_type == "medications":
        positives = feat_df.filter(
            (F.col("same_ndc_product") == 1.0) & (F.col("jw_drug") > 0.9)
        ).withColumn("label", F.lit(1.0))
        negatives = feat_df.filter(
            (F.col("jw_drug") < 0.5) & (F.col("token_overlap") < 0.2)
        ).limit(positives.count() * 3).withColumn("label", F.lit(0.0))
    else:
        positives = feat_df.filter(
            (F.col("same_icd_category") == 1.0) & (F.col("jw_title") > 0.92)
        ).withColumn("label", F.lit(1.0))
        negatives = feat_df.filter(
            (F.col("jw_title") < 0.4) & (F.col("token_overlap") < 0.1)
        ).limit(positives.count() * 3).withColumn("label", F.lit(0.0))

    return positives.union(negatives)


def train_and_evaluate(spark: SparkSession, entity_type: str, feature_cols: list[str]):
    print(f"\n=== ER Classifier: {entity_type} ===")

    gold_path = f"{GOLD_DIR}/{entity_type}"
    if os.path.exists(gold_path):
        labeled = spark.read.parquet(gold_path)
    else:
        print("  No manual gold set found — using heuristic labels")
        labeled = build_gold_labels(spark, entity_type)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
    rf = RandomForestClassifier(
        featuresCol="features_vec",
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        seed=42,
    )
    pipeline = Pipeline(stages=[assembler, rf])

    train, test = labeled.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)
    predictions = model.transform(test)

    # Evaluation
    bin_eval = BinaryClassificationEvaluator(labelCol="label")
    auc = bin_eval.evaluate(predictions)

    mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})

    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # Apply model to ALL candidate pairs to get same_as edges
    all_features = spark.read.parquet(f"{FEATURES_DIR}/{entity_type}")
    all_pred = model.transform(all_features)
    same_as = all_pred.filter(F.col("prediction") == 1.0)

    if entity_type == "medications":
        same_as = same_as.select(
            F.col("ndc_a").alias("id_a"),
            F.col("ndc_b").alias("id_b"),
            F.col("probability").alias("confidence"),
            F.lit("MEDICATION").alias("entity_type"),
        )
    else:
        same_as = same_as.select(
            F.col("code_a").alias("id_a"),
            F.col("code_b").alias("id_b"),
            F.col("probability").alias("confidence"),
            F.lit("DIAGNOSIS").alias("entity_type"),
        )

    same_as.write.mode("overwrite").parquet(f"{SAME_AS_OUT}/{entity_type}")
    print(f"  same_as edges: {same_as.count():,}")

    return {"entity_type": entity_type, "auc": auc, "precision": precision, "recall": recall, "f1": f1}


def main():
    spark = (
        SparkSession.builder
        .appName("KG-ERClassifier")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    results = []
    results.append(train_and_evaluate(spark, "medications", MED_FEATURE_COLS))
    results.append(train_and_evaluate(spark, "diagnoses", DIAG_FEATURE_COLS))

    # Save evaluation results
    import csv, os as _os
    _os.makedirs("/data/results", exist_ok=True)
    with open("/data/results/er_evaluation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_type", "auc", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(results)

    spark.stop()


if __name__ == "__main__":
    main()
