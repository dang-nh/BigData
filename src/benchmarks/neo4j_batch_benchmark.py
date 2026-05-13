"""
Optimization Experiment #4: Neo4j write batch size comparison.

Tests different batch sizes when loading Diagnosis nodes via neo4j-spark-connector.
Measures records/s and total time for each configuration.

Outputs results/graph_load_benchmark.csv.

Usage:
  spark-submit --packages org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3 \
    src/benchmarks/neo4j_batch_benchmark.py
"""

import os
import time
import csv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET_DIR = os.getenv("PARQUET_OUTPUT_DIR", "/data/parquet")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
RESULTS_DIR = "/data/results"

BATCH_SIZES = [500, 1000, 5000, 10000, 25000, 50000]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-BatchBenchmark")
        .getOrCreate()
    )


def clear_test_nodes(spark: SparkSession):
    """Remove test nodes between runs."""
    (
        spark.read.format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("query", "MATCH (n:BenchDiagnosis) DETACH DELETE n")
        .load()
    )


def write_batch(spark: SparkSession, df, batch_size: int) -> tuple[int, float]:
    total_rows = df.count()
    # Re-label as BenchDiagnosis to avoid touching prod data
    bench_df = df.withColumn("_label", F.lit("BenchDiagnosis"))

    t0 = time.time()
    (
        bench_df.write
        .format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("labels", ":BenchDiagnosis")
        .option("node.keys", "icd9_code")
        .option("batch.size", str(batch_size))
        .mode("overwrite")
        .save()
    )
    elapsed = time.time() - t0
    return total_rows, elapsed


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = spark.read.parquet(f"{PARQUET_DIR}/d_icd_diagnoses").cache()
    n_rows = df.count()
    print(f"Dataset: {n_rows:,} diagnosis records")

    results = []
    for batch_size in BATCH_SIZES:
        print(f"\nBatch size = {batch_size:,} ...")
        try:
            rows, elapsed = write_batch(spark, df, batch_size)
            rps = rows / max(elapsed, 0.001)
            print(f"  {rows:,} rows in {elapsed:.2f}s → {rps:,.0f} rec/s")
            results.append({
                "batch_size": batch_size,
                "rows": rows,
                "time_s": round(elapsed, 2),
                "records_per_s": round(rps, 0),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"batch_size": batch_size, "rows": n_rows,
                            "time_s": -1, "records_per_s": -1})

    out_path = f"{RESULTS_DIR}/graph_load_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "rows", "time_s", "records_per_s"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")

    # Cleanup
    try:
        with spark._jvm.org.neo4j.spark.DataSource:
            pass
    except Exception:
        pass

    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
