"""
Optimization Experiment #1: Spark skew handling.

Compares join performance on ADMISSIONS ⨝ NOTEEVENTS with and without
salting (key skew mitigation technique).

ADMISSIONS is skewed because some patients have many admissions but most
have very few — this causes data skew when joining by subject_id.

Outputs results/spark_skew.csv.

Usage:
  spark-submit src/benchmarks/spark_skew.py
"""

import os
import sys
import time
import csv
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

PARQUET_DIR = str(kg_paths.parquet_output_dir())
RESULTS_DIR = str(kg_paths.results_dir())
SALT_BUCKETS = 20


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-SkewBenchmark")
        .config("spark.sql.adaptive.enabled", "false")  # disable AQE to see raw skew
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def naive_join(admissions: DataFrame, notes: DataFrame) -> tuple[int, float]:
    """Plain join — subject_id skew will cause uneven task durations."""
    t0 = time.time()
    result = (
        admissions.alias("a")
        .join(notes.alias("n"), on="subject_id", how="inner")
        .select("a.hadm_id", "n.row_id", "n.category")
    )
    count = result.count()
    return count, time.time() - t0


def salted_join(admissions: DataFrame, notes: DataFrame, buckets: int) -> tuple[int, float]:
    """
    Salting: add a random bucket key to both sides.
    For the smaller side (admissions), replicate across all buckets.
    For the larger side (notes), assign one random bucket per row.
    """
    t0 = time.time()

    # Replicate admissions for each salt bucket
    adm_salted = admissions.crossJoin(
        admissions.sparkSession.range(buckets).toDF("salt")
    ).withColumn("join_key", F.concat(F.col("subject_id"), F.lit("_"), F.col("salt")))

    # Assign random bucket to notes
    notes_salted = notes.withColumn(
        "salt", (F.rand() * buckets).cast(IntegerType())
    ).withColumn("join_key", F.concat(F.col("subject_id"), F.lit("_"), F.col("salt")))

    result = (
        adm_salted.alias("a")
        .join(notes_salted.alias("n"), on="join_key", how="inner")
        .select("a.hadm_id", "n.row_id", "n.category")
    )
    count = result.count()
    return count, time.time() - t0


def measure_task_skew(spark: SparkSession, df: DataFrame, label: str) -> dict:
    """Measure max/min/mean task duration via Spark listener (approximated)."""
    # Trigger an action and measure stage metrics via SparkContext
    t0 = time.time()
    df.count()
    total = time.time() - t0
    # In a full implementation, hook SparkListener for per-task times.
    # Here we use wall clock as a proxy.
    return {"label": label, "wall_clock_s": round(total, 2)}


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data ...")
    admissions = spark.read.parquet(f"{PARQUET_DIR}/admissions").select("subject_id", "hadm_id")
    notes = spark.read.parquet(f"{PARQUET_DIR}/noteevents").select("subject_id", "row_id", "category")

    # Show skew distribution
    print("\nSubject_id frequency distribution (top 10):")
    (
        notes.groupBy("subject_id").count()
        .orderBy(F.col("count").desc())
        .show(10)
    )

    results = []

    # Run 3 times each to get stable measurements
    for run in range(1, 4):
        print(f"\n── Run {run}/3: Naive join ──")
        count, elapsed = naive_join(admissions, notes)
        print(f"  rows={count:,}  time={elapsed:.2f}s")
        results.append({"method": "naive", "run": run, "rows": count, "time_s": round(elapsed, 2)})

    for run in range(1, 4):
        print(f"\n── Run {run}/3: Salted join (buckets={SALT_BUCKETS}) ──")
        count, elapsed = salted_join(admissions, notes, SALT_BUCKETS)
        print(f"  rows={count:,}  time={elapsed:.2f}s")
        results.append({"method": f"salted_{SALT_BUCKETS}", "run": run, "rows": count, "time_s": round(elapsed, 2)})

    out_path = f"{RESULTS_DIR}/spark_skew.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "run", "rows", "time_s"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")

    # Summary
    naive_avg = sum(r["time_s"] for r in results if r["method"] == "naive") / 3
    salted_avg = sum(r["time_s"] for r in results if r["method"].startswith("salted")) / 3
    speedup = naive_avg / max(salted_avg, 0.001)
    print(f"\nSummary: naive_avg={naive_avg:.2f}s  salted_avg={salted_avg:.2f}s  speedup={speedup:.2f}x")

    spark.stop()


if __name__ == "__main__":
    main()
