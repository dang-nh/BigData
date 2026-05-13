"""
Optimization Experiment #5: Spark partition strategy comparison.

Compares different partitioning approaches for the ADMISSIONS ⨝ DIAGNOSES_ICD join:
  - default    : no explicit repartitioning (Spark default parallelism)
  - hash_key   : repartition(n, subject_id) — co-locate same patient
  - coalesce   : reduce partitions for small datasets
  - range      : rangePartition by subject_id (sorted)

Measures shuffle read bytes, number of tasks, and wall-clock time.

Outputs results/spark_partition.csv.

Usage:
  spark-submit src/benchmarks/partition_benchmark.py
"""

import os
import time
import csv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

PARQUET_DIR = os.getenv("PARQUET_OUTPUT_DIR", "/data/parquet")
RESULTS_DIR = "/data/results"


def get_spark(shuffle_partitions: int = 200) -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-PartitionBenchmark")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def run_aggregation(admissions: DataFrame, diagnoses: DataFrame, label: str) -> dict:
    """Join + group-by to count diagnoses per patient — representative of ETL workload."""
    t0 = time.time()

    result = (
        admissions.join(diagnoses, on="hadm_id", how="inner")
        .groupBy("subject_id")
        .agg(
            F.count("icd9_code").alias("total_diagnoses"),
            F.countDistinct("icd9_code").alias("unique_diagnoses"),
        )
    )
    row_count = result.count()
    elapsed = time.time() - t0

    # Try to get Spark metrics from last stage
    sc = admissions.sparkSession.sparkContext
    status = sc.statusTracker()
    stage_ids = status.getActiveStageIds() + status.getJobIdsForGroup(None)

    print(f"  [{label}] {row_count:,} patients, {elapsed:.2f}s")
    return {
        "strategy": label,
        "patients": row_count,
        "time_s": round(elapsed, 2),
    }


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    adm_raw = spark.read.parquet(f"{PARQUET_DIR}/admissions").select("subject_id", "hadm_id")
    dx_raw = spark.read.parquet(f"{PARQUET_DIR}/diagnoses_icd").select("hadm_id", "icd9_code")
    n_adm = adm_raw.count()
    print(f"Admissions: {n_adm:,}")

    results = []

    # ── Strategy 1: Default ──────────────────────────────────────────────────
    print("\n[1/4] Default partitioning ...")
    results.append(run_aggregation(adm_raw, dx_raw, "default"))

    # ── Strategy 2: Hash by subject_id ────────────────────────────────────────
    print("\n[2/4] Hash repartition by subject_id (200 partitions) ...")
    adm_hash = adm_raw.repartition(200, "subject_id")
    dx_hash = dx_raw.repartition(200, "hadm_id")
    results.append(run_aggregation(adm_hash, dx_hash, "hash_subject_id"))

    # ── Strategy 3: Smaller partition count ───────────────────────────────────
    print("\n[3/4] Coalesce to 50 partitions ...")
    adm_coal = adm_raw.coalesce(50)
    dx_coal = dx_raw.coalesce(50)
    results.append(run_aggregation(adm_coal, dx_coal, "coalesce_50"))

    # ── Strategy 4: Broadcast join (small diagnoses table) ────────────────────
    print("\n[4/4] Broadcast hint on admissions (small side) ...")
    results.append(run_aggregation(
        F.broadcast(adm_raw),
        dx_raw,
        "broadcast_admissions",
    ))

    out_path = f"{RESULTS_DIR}/spark_partition.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "patients", "time_s"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")
    for r in results:
        print(f"  {r['strategy']:25s}  {r['time_s']}s")

    spark.stop()


if __name__ == "__main__":
    main()
