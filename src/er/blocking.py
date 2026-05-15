"""
Entity Resolution — Phase 1: Blocking + MinHash LSH candidate generation.
Reduces O(n²) comparison space to manageable candidate pairs.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import MinHashLSH, Tokenizer, CountVectorizer, HashingTF, IDF

PARQUET_DIR = str(kg_paths.parquet_output_dir())
CANDIDATES_OUT = f"{PARQUET_DIR}/er_candidates"


# ── Blocking key UDFs ──────────────────────────────────────

@F.udf(StringType())
def drug_block_key(drug_name: str) -> str:
    """Blocking key for medications: first word + first char of second word."""
    if not drug_name:
        return "UNKNOWN"
    parts = drug_name.strip().upper().split()
    key = parts[0][:6] if parts else "UNKNOWN"
    return key


@F.udf(StringType())
def icd_chapter(icd_code: str) -> str:
    """ICD-9 chapter from code prefix (00-99)."""
    if not icd_code:
        return "UNK"
    code = icd_code.strip().lstrip("0")[:3]
    try:
        n = int(code)
        if n < 140:    return "INFECTIOUS"
        elif n < 240:  return "NEOPLASMS"
        elif n < 280:  return "ENDOCRINE"
        elif n < 290:  return "BLOOD"
        elif n < 320:  return "MENTAL"
        elif n < 390:  return "NERVOUS"
        elif n < 460:  return "CIRCULATORY"
        elif n < 520:  return "RESPIRATORY"
        elif n < 580:  return "DIGESTIVE"
        elif n < 630:  return "GENITOURINARY"
        elif n < 680:  return "SKIN"
        elif n < 740:  return "MUSCULOSKELETAL"
        elif n < 800:  return "CONGENITAL"
        else:          return "INJURY"
    except ValueError:
        return "OTHER"


def block_medications(spark: SparkSession) -> DataFrame:
    """Generate candidate medication pairs within same blocking key."""
    prescriptions = spark.read.parquet(f"{PARQUET_DIR}/prescriptions")

    drugs = (
        prescriptions
        .select("ndc", "drug", "drug_name_poe", "drug_name_generic")
        .filter(F.col("ndc").isNotNull())
        .dropDuplicates(["ndc"])
        .withColumn("block_key", drug_block_key(F.col("drug")))
        .withColumn("drug_tokens", F.split(F.upper(F.col("drug")), r"\s+"))
    )

    # MinHash LSH on drug name token sets
    htf = HashingTF(inputCol="drug_tokens", outputCol="features", numFeatures=256)
    drugs_feat = htf.transform(drugs)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(drugs_feat)
    similar = model.approxSimilarityJoin(
        drugs_feat, drugs_feat,
        threshold=0.4,
        distCol="jaccard_dist"
    ).filter("datasetA.ndc < datasetB.ndc")  # avoid self + duplicates

    return similar.select(
        F.col("datasetA.ndc").alias("ndc_a"),
        F.col("datasetB.ndc").alias("ndc_b"),
        F.col("datasetA.drug").alias("drug_a"),
        F.col("datasetB.drug").alias("drug_b"),
        F.col("jaccard_dist"),
        F.lit("MEDICATION").alias("entity_type"),
    )


def block_diagnoses(spark: SparkSession) -> DataFrame:
    """Generate candidate diagnosis pairs within same ICD chapter."""
    d_icd = spark.read.parquet(f"{PARQUET_DIR}/d_icd_diagnoses")

    diags = (
        d_icd
        .filter(F.col("icd9_code").isNotNull())
        .withColumn("chapter", icd_chapter(F.col("icd9_code")))
        .withColumn("title_tokens", F.split(F.upper(F.col("short_title")), r"\s+"))
    )

    htf = HashingTF(inputCol="title_tokens", outputCol="features", numFeatures=256)
    diags_feat = htf.transform(diags)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(diags_feat)
    similar = model.approxSimilarityJoin(
        diags_feat, diags_feat,
        threshold=0.3,
        distCol="jaccard_dist"
    ).filter("datasetA.icd9_code < datasetB.icd9_code")

    return similar.select(
        F.col("datasetA.icd9_code").alias("code_a"),
        F.col("datasetB.icd9_code").alias("code_b"),
        F.col("datasetA.short_title").alias("title_a"),
        F.col("datasetB.short_title").alias("title_b"),
        F.col("jaccard_dist"),
        F.lit("DIAGNOSIS").alias("entity_type"),
    )


def main():
    spark = (
        SparkSession.builder
        .appName("KG-ERBlocking")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    med_candidates = block_medications(spark)
    diag_candidates = block_diagnoses(spark)

    med_candidates.write.mode("overwrite").parquet(f"{CANDIDATES_OUT}/medications")
    diag_candidates.write.mode("overwrite").parquet(f"{CANDIDATES_OUT}/diagnoses")

    print(f"Medication candidate pairs: {med_candidates.count():,}")
    print(f"Diagnosis candidate pairs:  {diag_candidates.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()
