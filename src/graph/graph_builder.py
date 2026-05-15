"""
Build Neo4j Knowledge Graph from Parquet data using neo4j-spark-connector.
Loads nodes then relationships in dependency order.
"""

import os
import sys
import time
import csv as csv_mod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

PARQUET_DIR = str(kg_paths.parquet_output_dir())
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

BATCH_SIZE = int(os.getenv("NEO4J_BATCH_SIZE", "10000"))


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-GraphBuilder")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def neo4j_write(df: DataFrame, labels: str, node_keys: list[str] | None = None,
                rel_type: str | None = None, batch_size: int = BATCH_SIZE):
    """Write DataFrame to Neo4j via connector."""
    writer = (
        df.write
        .format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("batch.size", str(batch_size))
        .mode("overwrite")
    )

    if rel_type:
        writer = writer.option("relationship", rel_type)
    else:
        writer = writer.option("labels", labels)
        if node_keys:
            writer = writer.option("node.keys", ",".join(node_keys))

    writer.save()


def load_patients(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/patients")
        .select("subject_id", "gender", "dob", "dod", "expire_flag")
        .withColumn("dob", F.col("dob").cast("string"))
        .withColumn("dod", F.col("dod").cast("string"))
    )
    neo4j_write(df, ":Patient", node_keys=["subject_id"])
    print(f"  Patient nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_admissions(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/admissions")
        .select("hadm_id", "subject_id", "admittime", "dischtime",
                "admission_type", "diagnosis", "hospital_expire_flag")
        .withColumn("admittime", F.col("admittime").cast("string"))
        .withColumn("dischtime", F.col("dischtime").cast("string"))
    )
    neo4j_write(df, ":Admission", node_keys=["hadm_id"])
    print(f"  Admission nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_diagnoses(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/d_icd_diagnoses")
        .select("icd9_code", "short_title", "long_title")
    )
    neo4j_write(df, ":Diagnosis", node_keys=["icd9_code"])
    print(f"  Diagnosis nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_medications(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/prescriptions")
        .select("ndc", "drug", "drug_name_poe", "drug_name_generic", "formulary_drug_cd")
        .filter(F.col("ndc").isNotNull())
        .dropDuplicates(["ndc"])
    )
    neo4j_write(df, ":Medication", node_keys=["ndc"])
    print(f"  Medication nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_labtests(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/d_labitems")
        .select("itemid", "label", "fluid", "category", "loinc_code")
    )
    neo4j_write(df, ":LabTest", node_keys=["itemid"])
    print(f"  LabTest nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_concepts(spark: SparkSession):
    t0 = time.time()
    df = (
        spark.read.parquet(f"{PARQUET_DIR}/concepts")
        .select("cui", "name")
        .dropDuplicates(["cui"])
        .filter(F.col("cui") != "UNKNOWN")
    )
    neo4j_write(df, ":Concept", node_keys=["cui"])
    print(f"  Concept nodes: {df.count():,} ({time.time()-t0:.1f}s)")


def load_relationships(spark: SparkSession):
    # Patient -[:HAS_ADMISSION]-> Admission
    t0 = time.time()
    adm = spark.read.parquet(f"{PARQUET_DIR}/admissions").select("subject_id", "hadm_id")
    (
        adm
        .select(
            F.struct(F.lit("Patient"), F.col("subject_id")).alias("source"),
            F.struct(F.lit("Admission"), F.col("hadm_id")).alias("target"),
        )
        .write.format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("relationship", "HAS_ADMISSION")
        .option("relationship.source.labels", ":Patient")
        .option("relationship.source.node.keys", "subject_id")
        .option("relationship.target.labels", ":Admission")
        .option("relationship.target.node.keys", "hadm_id")
        .option("batch.size", str(BATCH_SIZE))
        .mode("overwrite")
        .save()
    )
    print(f"  HAS_ADMISSION edges: {adm.count():,} ({time.time()-t0:.1f}s)")

    # Admission -[:HAS_DIAGNOSIS {seq_num}]-> Diagnosis
    t0 = time.time()
    diag_links = (
        spark.read.parquet(f"{PARQUET_DIR}/diagnoses_icd")
        .select("hadm_id", "icd9_code", "seq_num")
        .filter(F.col("icd9_code").isNotNull())
    )
    (
        diag_links
        .write.format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("relationship", "HAS_DIAGNOSIS")
        .option("relationship.properties", "seq_num")
        .option("relationship.source.labels", ":Admission")
        .option("relationship.source.node.keys", "hadm_id")
        .option("relationship.target.labels", ":Diagnosis")
        .option("relationship.target.node.keys", "icd9_code")
        .option("batch.size", str(BATCH_SIZE))
        .mode("overwrite")
        .save()
    )
    print(f"  HAS_DIAGNOSIS edges: {diag_links.count():,} ({time.time()-t0:.1f}s)")

    # Admission -[:PRESCRIBED]-> Medication
    t0 = time.time()
    rx_links = (
        spark.read.parquet(f"{PARQUET_DIR}/prescriptions")
        .select("hadm_id", "ndc", "dose_val_rx", "dose_unit_rx", "route")
        .filter(F.col("ndc").isNotNull())
    )
    (
        rx_links
        .write.format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("relationship", "PRESCRIBED")
        .option("relationship.properties", "dose_val_rx,dose_unit_rx,route")
        .option("relationship.source.labels", ":Admission")
        .option("relationship.source.node.keys", "hadm_id")
        .option("relationship.target.labels", ":Medication")
        .option("relationship.target.node.keys", "ndc")
        .option("batch.size", str(BATCH_SIZE))
        .mode("overwrite")
        .save()
    )
    print(f"  PRESCRIBED edges: {rx_links.count():,} ({time.time()-t0:.1f}s)")

    # ClinicalNote -[:MENTIONS]-> Concept
    t0 = time.time()
    mention_links = (
        spark.read.parquet(f"{PARQUET_DIR}/concepts")
        .select("note_row_id", "cui", "score")
        .filter(F.col("cui") != "UNKNOWN")
    )
    (
        mention_links
        .write.format("org.neo4j.spark.DataSource")
        .option("url", NEO4J_URI)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", NEO4J_USER)
        .option("authentication.basic.password", NEO4J_PASS)
        .option("relationship", "MENTIONS")
        .option("relationship.properties", "score")
        .option("relationship.source.labels", ":ClinicalNote")
        .option("relationship.source.node.keys", "note_row_id")
        .option("relationship.target.labels", ":Concept")
        .option("relationship.target.node.keys", "cui")
        .option("batch.size", str(BATCH_SIZE))
        .mode("overwrite")
        .save()
    )
    print(f"  MENTIONS edges: {mention_links.count():,} ({time.time()-t0:.1f}s)")


def load_same_as_edges(spark: SparkSession):
    """Load ER-resolved same_as edges into the graph."""
    for entity_type in ["medications", "diagnoses"]:
        path = f"{PARQUET_DIR}/same_as_edges/{entity_type}"
        if not os.path.exists(path):
            continue
        t0 = time.time()
        df = spark.read.parquet(path)

        label = ":Medication" if entity_type == "medications" else ":Diagnosis"
        key = "ndc" if entity_type == "medications" else "icd9_code"

        df_renamed = df.withColumnRenamed("id_a", key + "_a").withColumnRenamed("id_b", key + "_b")

        (
            df_renamed
            .write.format("org.neo4j.spark.DataSource")
            .option("url", NEO4J_URI)
            .option("authentication.type", "basic")
            .option("authentication.basic.username", NEO4J_USER)
            .option("authentication.basic.password", NEO4J_PASS)
            .option("relationship", "SAME_AS")
            .option("relationship.source.labels", label)
            .option("relationship.source.node.keys", f"{key}_a:{key}")
            .option("relationship.target.labels", label)
            .option("relationship.target.node.keys", f"{key}_b:{key}")
            .option("batch.size", str(BATCH_SIZE))
            .mode("overwrite")
            .save()
        )
        print(f"  SAME_AS edges ({entity_type}): {df.count():,} ({time.time()-t0:.1f}s)")


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("=== Loading nodes ===")
    load_patients(spark)
    load_admissions(spark)
    load_diagnoses(spark)
    load_medications(spark)
    load_labtests(spark)
    load_concepts(spark)

    print("\n=== Loading relationships ===")
    load_relationships(spark)

    print("\n=== Loading same_as (ER) edges ===")
    load_same_as_edges(spark)

    print("\nKnowledge Graph build complete.")
    spark.stop()


if __name__ == "__main__":
    main()
