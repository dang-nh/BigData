"""
Simulate streaming clinical notes: read NOTEEVENTS.csv, publish to Kafka
ordered by charttime (realtime simulation with configurable speed factor).

Usage:
  python src/ingest/kafka_producer.py [--speed 100] [--topic clinical-notes]
"""

import os
import csv
import json
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

from kafka import KafkaProducer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def make_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=5,
        batch_size=16384,
        linger_ms=10,
    )


def stream_notes(
    csv_path: str,
    topic: str,
    bootstrap_servers: str,
    speed_factor: float = 100.0,
    limit: int = 0,
):
    producer = make_producer(bootstrap_servers)
    log.info(f"Streaming {csv_path} -> topic '{topic}' at {speed_factor}x realtime")

    sent = 0
    prev_sim_time = None
    prev_wall_time = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            charttime_str = row.get("charttime") or row.get("chartdate") or ""
            if charttime_str:
                try:
                    sim_time = datetime.strptime(charttime_str[:19], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        sim_time = datetime.strptime(charttime_str[:10], "%Y-%m-%d")
                    except ValueError:
                        sim_time = None

                if sim_time and prev_sim_time and prev_wall_time:
                    sim_delta = (sim_time - prev_sim_time).total_seconds()
                    wall_delta = sim_delta / speed_factor
                    elapsed = time.time() - prev_wall_time
                    sleep_time = wall_delta - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                prev_sim_time = sim_time
                prev_wall_time = time.time()

            payload = {
                "row_id": row.get("row_id"),
                "subject_id": row.get("subject_id"),
                "hadm_id": row.get("hadm_id"),
                "charttime": charttime_str,
                "category": row.get("category", ""),
                "description": row.get("description", ""),
                "text": row.get("text", ""),
            }

            producer.send(topic, key=row.get("subject_id"), value=payload)
            sent += 1

            if sent % 1000 == 0:
                log.info(f"  Sent {sent:,} notes ...")

            if limit and sent >= limit:
                break

    producer.flush()
    producer.close()
    log.info(f"Done. Total sent: {sent:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(kg_paths.mimic_data_dir() / "NOTEEVENTS.csv"),
        help="Path to NOTEEVENTS.csv",
    )
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_NOTES", "clinical-notes"))
    parser.add_argument("--servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--speed", type=float, default=100.0, help="Simulation speed multiplier")
    parser.add_argument("--limit", type=int, default=0, help="Max notes to send (0 = all)")
    args = parser.parse_args()

    stream_notes(args.csv, args.topic, args.servers, args.speed, args.limit)


if __name__ == "__main__":
    main()
