"""
Phase 7.2 — Scalability test: run pipeline at 10%, 50%, 100% data sizes.

Measures ingestion time, ER candidate generation time, and graph build time
for each data fraction.

Outputs results/scalability.csv.

Usage:
  spark-submit src/benchmarks/scalability.py
"""

import os
import sys
import time
import csv
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import MinHashLSH, HashingTF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

PARQUET_DIR = str(kg_paths.parquet_output_dir())
RESULTS_DIR = str(kg_paths.results_dir())

DATA_FRACTIONS = [0.1, 0.5, 1.0]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-Scalability")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def measure_ingest(spark: SparkSession, fraction: float) -> dict:
    """Measure Parquet read + schema validation time."""
    t0 = time.time()
    tables = ["patients", "admissions", "diagnoses_icd", "prescriptions", "noteevents"]
    total_rows = 0
    for table in tables:
        df = spark.read.parquet(f"{PARQUET_DIR}/{table}")
        if fraction < 1.0:
            df = df.sample(fraction=fraction, seed=42)
        total_rows += df.count()
    elapsed = time.time() - t0
    return {"phase": "ingest", "fraction": fraction, "rows": total_rows,
            "time_s": round(elapsed, 2), "throughput_rps": round(total_rows / elapsed, 0)}


def measure_er(spark: SparkSession, fraction: float) -> dict:
    """Measure ER candidate generation time (LSH blocking)."""
    t0 = time.time()
    df = spark.read.parquet(f"{PARQUET_DIR}/prescriptions")
    if fraction < 1.0:
        df = df.sample(fraction=fraction, seed=42)

    drugs = (
        df.select("ndc", "drug")
        .filter(F.col("ndc").isNotNull())
        .dropDuplicates(["ndc"])
        .withColumn("tokens", F.split(F.upper(F.col("drug")), r"\s+"))
    )
    htf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=256)
    drugs_feat = htf.transform(drugs)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(drugs_feat)
    candidates = (
        model.approxSimilarityJoin(drugs_feat, drugs_feat, threshold=0.4, distCol="dist")
        .filter(F.col("datasetA.ndc") < F.col("datasetB.ndc"))
    )
    candidate_count = candidates.count()
    elapsed = time.time() - t0

    n_drugs = drugs.count()
    return {"phase": "er", "fraction": fraction, "rows": n_drugs,
            "candidates": candidate_count, "time_s": round(elapsed, 2),
            "throughput_rps": round(n_drugs / elapsed, 0)}


def measure_graph_build(spark: SparkSession, fraction: float) -> dict:
    """
    Measure graph-load preparation time (ETL side).
    Actual Neo4j write excluded (network-bound); measures Spark transformations.
    """
    t0 = time.time()
    adm = spark.read.parquet(f"{PARQUET_DIR}/admissions")
    dx = spark.read.parquet(f"{PARQUET_DIR}/diagnoses_icd")
    rx = spark.read.parquet(f"{PARQUET_DIR}/prescriptions")

    if fraction < 1.0:
        adm = adm.sample(fraction=fraction, seed=42)
        dx = dx.sample(fraction=fraction, seed=42)
        rx = rx.sample(fraction=fraction, seed=42)

    # Simulate graph preparation: join and deduplicate
    node_adm = adm.select("hadm_id", "subject_id").dropDuplicates()
    edge_dx = dx.join(node_adm.select("hadm_id"), on="hadm_id", how="inner")
    edge_rx = rx.join(node_adm.select("hadm_id"), on="hadm_id", how="inner").filter(F.col("ndc").isNotNull())

    total = node_adm.count() + edge_dx.count() + edge_rx.count()
    elapsed = time.time() - t0

    return {"phase": "graph_prep", "fraction": fraction, "rows": total,
            "time_s": round(elapsed, 2), "throughput_rps": round(total / elapsed, 0)}


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for fraction in DATA_FRACTIONS:
        pct = int(fraction * 100)
        print(f"\n{'='*50}")
        print(f"  Data fraction: {pct}%")
        print(f"{'='*50}")

        print(f"  [1/3] Ingest benchmark ...")
        ingest = measure_ingest(spark, fraction)
        print(f"    rows={ingest['rows']:,}  time={ingest['time_s']}s  {ingest['throughput_rps']} rec/s")
        results.append(ingest)

        print(f"  [2/3] ER benchmark ...")
        er = measure_er(spark, fraction)
        print(f"    drugs={er['rows']:,}  candidates={er['candidates']:,}  time={er['time_s']}s")
        results.append(er)

        print(f"  [3/3] Graph prep benchmark ...")
        gp = measure_graph_build(spark, fraction)
        print(f"    rows={gp['rows']:,}  time={gp['time_s']}s")
        results.append(gp)

    out_path = f"{RESULTS_DIR}/scalability.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "fraction", "rows",
                                               "time_s", "throughput_rps", "candidates"])
        writer.writeheader()
        for r in results:
            r.setdefault("candidates", "")
            writer.writerow(r)

    print(f"\nScalability results saved to {out_path}")
    spark.stop()


if __name__ == "__main__":
    main()
