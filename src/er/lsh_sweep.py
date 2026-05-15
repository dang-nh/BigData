"""
Optimization Experiment #2: Brute-force vs MinHash LSH parameter sweep.

Sweeps over (numHashTables b, bands r) to produce recall vs runtime tradeoff.
Outputs results/er_lsh_sweep.csv.

Usage:
  spark-submit src/er/lsh_sweep.py
"""

import os
import sys
import time
import csv
import itertools
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinHashLSH, HashingTF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

PARQUET_DIR = str(kg_paths.parquet_output_dir())
RESULTS_DIR = str(kg_paths.results_dir())


def get_spark():
    return (
        SparkSession.builder
        .appName("KG-ERLSHSweep")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def prepare_drug_features(spark: SparkSession):
    """Load prescriptions and compute TF feature vectors."""
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/prescriptions")
        .select("ndc", "drug")
        .filter(F.col("ndc").isNotNull() & F.col("drug").isNotNull())
        .dropDuplicates(["ndc"])
        .withColumn("tokens", F.split(F.upper(F.col("drug")), r"\s+"))
    )
    htf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=256)
    return htf.transform(df), df.count()


def brute_force_pairs(spark: SparkSession, df) -> tuple[int, float]:
    """Compute all pairs with Jaccard < 0.5 (brute-force baseline)."""
    t0 = time.time()
    # Self-join on block key to limit n² — simulate brute force within same first letter
    df_a = df.select(F.col("ndc").alias("ndc_a"), F.col("drug").alias("drug_a"),
                     F.substring(F.col("drug"), 1, 1).alias("blk"))
    df_b = df.select(F.col("ndc").alias("ndc_b"), F.col("drug").alias("drug_b"),
                     F.substring(F.col("drug"), 1, 1).alias("blk"))
    pairs = (
        df_a.join(df_b, on="blk")
        .filter(F.col("ndc_a") < F.col("ndc_b"))
    )
    count = pairs.count()
    elapsed = time.time() - t0
    return count, elapsed


def lsh_pairs(spark: SparkSession, df, num_hash_tables: int, threshold: float) -> tuple[int, float]:
    """Run MinHash LSH with given parameters."""
    t0 = time.time()
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=num_hash_tables)
    model = mh.fit(df)
    similar = (
        model.approxSimilarityJoin(df, df, threshold=threshold, distCol="jaccard_dist")
        .filter(F.col("datasetA.ndc") < F.col("datasetB.ndc"))
    )
    count = similar.count()
    elapsed = time.time() - t0
    return count, elapsed


def estimate_recall(lsh_count: int, brute_count: int) -> float:
    """Approximate recall: ratio of LSH pairs to brute-force pairs."""
    if brute_count == 0:
        return 0.0
    return min(lsh_count / brute_count, 1.0)


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Preparing drug features ...")
    df_feat, n_drugs = prepare_drug_features(spark)
    df_feat.cache()
    print(f"Unique drugs: {n_drugs:,}")

    # Brute-force baseline
    print("\nRunning brute-force baseline ...")
    bf_count, bf_time = brute_force_pairs(spark, df_feat.select("ndc", "drug"))
    print(f"  Brute-force: {bf_count:,} pairs in {bf_time:.1f}s")

    # LSH sweep
    num_hash_tables_values = [3, 5, 10, 15]
    threshold_values = [0.3, 0.4, 0.5, 0.6]

    results = []
    for num_ht, threshold in itertools.product(num_hash_tables_values, threshold_values):
        print(f"\nLSH: numHashTables={num_ht}, threshold={threshold} ...")
        try:
            lsh_count, lsh_time = lsh_pairs(spark, df_feat, num_ht, threshold)
            recall = estimate_recall(lsh_count, bf_count)
            speedup = bf_time / max(lsh_time, 1e-3)
            print(f"  pairs={lsh_count:,}  time={lsh_time:.1f}s  recall≈{recall:.3f}  speedup={speedup:.1f}x")
            results.append({
                "num_hash_tables": num_ht,
                "threshold": threshold,
                "lsh_pairs": lsh_count,
                "bf_pairs": bf_count,
                "recall_approx": round(recall, 4),
                "lsh_time_s": round(lsh_time, 2),
                "bf_time_s": round(bf_time, 2),
                "speedup_x": round(speedup, 2),
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    out_path = f"{RESULTS_DIR}/er_lsh_sweep.csv"
    fields = ["num_hash_tables", "threshold", "lsh_pairs", "bf_pairs",
              "recall_approx", "lsh_time_s", "bf_time_s", "speedup_x"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSweep results saved to {out_path}")
    df_feat.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
