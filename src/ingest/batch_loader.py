"""
Batch ingest MIMIC-III CSV files -> Parquet (partitioned by subject_id).
Run via: spark-submit src/ingest/batch_loader.py
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIMIC_DIR = str(kg_paths.mimic_data_dir())
OUT_DIR = str(kg_paths.parquet_output_dir())

# Each entry: (csv_filename, partition_by_col or None)
TABLES = [
    ("PATIENTS.csv",        "SUBJECT_ID"),
    ("ADMISSIONS.csv",      "SUBJECT_ID"),
    ("DIAGNOSES_ICD.csv",   "SUBJECT_ID"),
    ("PROCEDURES_ICD.csv",  "SUBJECT_ID"),
    ("PRESCRIPTIONS.csv",   "SUBJECT_ID"),
    ("LABEVENTS.csv",       "SUBJECT_ID"),
    ("NOTEEVENTS.csv",      "SUBJECT_ID"),
    ("CAREGIVERS.csv",      None),
    ("D_ICD_DIAGNOSES.csv", None),
    ("D_ICD_PROCEDURES.csv", None),
    ("D_LABITEMS.csv",      None),
]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("MIMIC-BatchIngest")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def load_and_save(spark: SparkSession, csv_file: str, partition_col: str | None):
    table_name = csv_file.replace(".csv", "").lower()
    input_path = os.path.join(MIMIC_DIR, csv_file)
    output_path = os.path.join(OUT_DIR, table_name)

    log.info(f"Loading {csv_file} ...")
    t0 = time.time()

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("nullValue", "")
        .csv(input_path)
    )

    # Normalize column names to lowercase
    df = df.toDF(*[c.lower() for c in df.columns])

    total = df.count()
    elapsed = time.time() - t0
    log.info(f"  {table_name}: {total:,} rows loaded in {elapsed:.1f}s ({total/elapsed:,.0f} rec/s)")

    # Log null counts for quality check
    null_counts = {c: df.filter(F.col(c).isNull()).count() for c in df.columns}
    high_null = {c: v for c, v in null_counts.items() if v / max(total, 1) > 0.1}
    if high_null:
        log.warning(f"  High null columns (>10%): {high_null}")

    write_op = df.write.mode("overwrite").parquet
    if partition_col and partition_col.lower() in df.columns:
        df = df.repartition(partition_col.lower())
        df.write.mode("overwrite").partitionBy(partition_col.lower()).parquet(output_path)
    else:
        df.write.mode("overwrite").parquet(output_path)

    log.info(f"  Saved to {output_path}")
    return {"table": table_name, "rows": total, "elapsed_s": round(elapsed, 2)}


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    results = []
    for csv_file, partition_col in TABLES:
        input_path = os.path.join(MIMIC_DIR, csv_file)
        if not os.path.exists(input_path):
            log.warning(f"  Skipping {csv_file} — not found at {input_path}")
            continue
        result = load_and_save(spark, csv_file, partition_col)
        results.append(result)

    # Save benchmark
    import csv
    res = kg_paths.results_dir()
    res.mkdir(parents=True, exist_ok=True)
    out_csv = str(res / "ingestion_benchmark.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["table", "rows", "elapsed_s"])
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Benchmark saved to {out_csv}")
    spark.stop()


if __name__ == "__main__":
    main()
