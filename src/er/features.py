"""
ER feature engineering: compute pairwise similarity features for candidate pairs.
"""

import os
import sys
import jellyfish
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType

PARQUET_DIR = str(kg_paths.parquet_output_dir())
CANDIDATES_DIR = f"{PARQUET_DIR}/er_candidates"
FEATURES_OUT = f"{PARQUET_DIR}/er_features"


# ── String similarity UDFs ──────────────────────────────────

@F.udf(FloatType())
def jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(jellyfish.jaro_winkler_similarity(a.upper(), b.upper()))


@F.udf(FloatType())
def norm_levenshtein(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = jellyfish.levenshtein_distance(a.upper(), b.upper())
    max_len = max(len(a), len(b), 1)
    return float(1 - dist / max_len)


@F.udf(FloatType())
def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word token sets."""
    if not a or not b:
        return 0.0
    sa = set(a.upper().split())
    sb = set(b.upper().split())
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def compute_medication_features(spark: SparkSession) -> DataFrame:
    candidates = spark.read.parquet(f"{CANDIDATES_DIR}/medications")

    return candidates.withColumn(
        "jw_drug", jaro_winkler(F.col("drug_a"), F.col("drug_b"))
    ).withColumn(
        "lev_drug", norm_levenshtein(F.col("drug_a"), F.col("drug_b"))
    ).withColumn(
        "token_overlap", token_overlap(F.col("drug_a"), F.col("drug_b"))
    ).withColumn(
        "jaccard_sim", F.lit(1.0) - F.col("jaccard_dist")
    ).withColumn(
        # Hard positive: same NDC prefix (first 9 digits = labeler + product)
        "same_ndc_product", (
            F.substring(F.regexp_replace(F.col("ndc_a"), "-", ""), 1, 9) ==
            F.substring(F.regexp_replace(F.col("ndc_b"), "-", ""), 1, 9)
        ).cast(FloatType())
    ).select(
        "ndc_a", "ndc_b",
        "jw_drug", "lev_drug", "token_overlap", "jaccard_sim", "same_ndc_product",
        F.lit(None).cast(FloatType()).alias("label"),  # filled by gold set
    )


def compute_diagnosis_features(spark: SparkSession) -> DataFrame:
    candidates = spark.read.parquet(f"{CANDIDATES_DIR}/diagnoses")

    return candidates.withColumn(
        "jw_title", jaro_winkler(F.col("title_a"), F.col("title_b"))
    ).withColumn(
        "lev_title", norm_levenshtein(F.col("title_a"), F.col("title_b"))
    ).withColumn(
        "token_overlap", token_overlap(F.col("title_a"), F.col("title_b"))
    ).withColumn(
        "jaccard_sim", F.lit(1.0) - F.col("jaccard_dist")
    ).withColumn(
        # Same first 3 digits = same ICD-9 category (coarser grouping)
        "same_icd_category", (
            F.substring("code_a", 1, 3) == F.substring("code_b", 1, 3)
        ).cast(FloatType())
    ).select(
        "code_a", "code_b",
        "jw_title", "lev_title", "token_overlap", "jaccard_sim", "same_icd_category",
        F.lit(None).cast(FloatType()).alias("label"),
    )


def main():
    spark = (
        SparkSession.builder
        .appName("KG-ERFeatures")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    med_feat = compute_medication_features(spark)
    med_feat.write.mode("overwrite").parquet(f"{FEATURES_OUT}/medications")

    diag_feat = compute_diagnosis_features(spark)
    diag_feat.write.mode("overwrite").parquet(f"{FEATURES_OUT}/diagnoses")

    print(f"Medication feature rows: {med_feat.count():,}")
    print(f"Diagnosis feature rows:  {diag_feat.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()
