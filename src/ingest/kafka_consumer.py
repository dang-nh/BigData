"""
Spark Structured Streaming: consume clinical notes from Kafka -> MongoDB.
Run via: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,...
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_NOTES", "clinical-notes")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin123@mongodb:27017")
MONGO_DB = os.getenv("MONGO_DB", "mimic_kg")
CHECKPOINT_DIR = str(kg_paths.checkpoints_notes_stream_dir())

NOTE_SCHEMA = StructType([
    StructField("row_id", StringType()),
    StructField("subject_id", StringType()),
    StructField("hadm_id", StringType()),
    StructField("charttime", StringType()),
    StructField("category", StringType()),
    StructField("description", StringType()),
    StructField("text", StringType()),
])


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("KG-NoteStream")
        .config("spark.mongodb.output.uri", f"{MONGO_URI}/{MONGO_DB}.clinical_notes")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def write_to_mongo(batch_df, batch_id: int):
    """Write each micro-batch to MongoDB."""
    if batch_df.isEmpty():
        return
    (
        batch_df
        .write
        .format("mongodb")
        .mode("append")
        .option("database", MONGO_DB)
        .option("collection", "clinical_notes")
        .save()
    )


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 5000)
        .load()
    )

    notes_df = (
        raw_stream
        .select(F.from_json(F.col("value").cast("string"), NOTE_SCHEMA).alias("data"))
        .select("data.*")
        .withColumn("ingested_at", F.current_timestamp())
    )

    query = (
        notes_df.writeStream
        .foreachBatch(write_to_mongo)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="10 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
