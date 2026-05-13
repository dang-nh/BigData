"""
NLP concept extraction from clinical notes using scispaCy + UMLS linker.
Runs as a Spark job; uses mapPartitions to load model once per partition.
"""

import os
import json
from typing import Iterator
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, ArrayType
)

PARQUET_DIR = os.getenv("PARQUET_OUTPUT_DIR", "/data/parquet")
OUT_DIR = os.getenv("PARQUET_OUTPUT_DIR", "/data/parquet") + "/concepts"

CONCEPT_SCHEMA = StructType([
    StructField("note_row_id", StringType()),
    StructField("subject_id", StringType()),
    StructField("hadm_id", StringType()),
    StructField("cui", StringType()),
    StructField("name", StringType()),
    StructField("score", FloatType()),
    StructField("char_start", IntegerType()),
    StructField("char_end", IntegerType()),
    StructField("entity_text", StringType()),
])


def extract_concepts_partition(rows: Iterator[Row]) -> Iterator[Row]:
    """Load scispaCy model once per executor partition, then process all rows."""
    import spacy

    nlp = spacy.load("en_core_sci_md")
    # Add UMLS entity linker — requires scispacy[umls] and a downloaded KB
    try:
        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.85,
        })
        has_linker = True
    except Exception:
        has_linker = False

    for row in rows:
        text = row.get("text") or ""
        note_id = row.get("row_id", "")
        subj_id = row.get("subject_id", "")
        hadm_id = row.get("hadm_id", "")

        if not text.strip():
            continue

        # Truncate to first 10k chars to keep latency bounded
        doc = nlp(text[:10000])

        for ent in doc.ents:
            if has_linker and ent._.kb_ents:
                for cui, score in ent._.kb_ents[:3]:  # top-3 candidates
                    yield Row(
                        note_row_id=str(note_id),
                        subject_id=str(subj_id),
                        hadm_id=str(hadm_id),
                        cui=cui,
                        name=ent._.kb_ents[0][0] if ent._.kb_ents else ent.text,
                        score=float(score),
                        char_start=ent.start_char,
                        char_end=ent.end_char,
                        entity_text=ent.text,
                    )
            else:
                # Fallback: no UMLS CUI, use entity text
                yield Row(
                    note_row_id=str(note_id),
                    subject_id=str(subj_id),
                    hadm_id=str(hadm_id),
                    cui="UNKNOWN",
                    name=ent.text,
                    score=1.0,
                    char_start=ent.start_char,
                    char_end=ent.end_char,
                    entity_text=ent.text,
                )


def main():
    spark = (
        SparkSession.builder
        .appName("KG-ConceptExtraction")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    notes_df = spark.read.parquet(f"{PARQUET_DIR}/noteevents")

    # Sample for dev; remove filter for full run
    sample = os.getenv("NLP_SAMPLE_NOTES", "0")
    if sample != "0":
        notes_df = notes_df.limit(int(sample))

    concepts_rdd = notes_df.rdd.mapPartitions(extract_concepts_partition)
    concepts_df = spark.createDataFrame(concepts_rdd, schema=CONCEPT_SCHEMA)

    concepts_df.write.mode("overwrite").parquet(OUT_DIR)
    total = concepts_df.count()
    print(f"Extracted {total:,} concept mentions -> {OUT_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
