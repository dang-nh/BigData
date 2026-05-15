# Knowledge Graph Integration for Medical Diagnosis Support

A Big Data pipeline that integrates heterogeneous MIMIC-III clinical data into a unified Knowledge Graph, enabling cohort discovery, differential diagnosis assistance, and medication analytics.

## Architecture

```
MIMIC-III CSVs ──► Spark ETL ──► Parquet (HDFS)
                                      │
NOTEEVENTS ──► Kafka ──► Spark SS ──► MongoDB
                                      │
                              Spark NLP (scispaCy)
                                      │
                              Entity Resolution (LSH)
                                      │
                              Neo4j Knowledge Graph
                                      │
                    ┌─────────────────┼──────────────────┐
               Cypher Queries    GDS Analytics      Dashboard
```

## Stack

| Layer | Technology |
|---|---|
| Streaming | Apache Kafka 3.5 |
| Batch Processing | Apache Spark 3.5 |
| Document Store | MongoDB 7.0 |
| Graph DB | Neo4j 5.15 + GDS + APOC |
| Full-text Search | Elasticsearch 8.11 |
| NLP | scispaCy + UMLS linker |
| Entity Resolution | Spark MLlib + MinHash LSH |

## Quick Start

```bash
# 1. Copy env config
cp .env.example .env
# Edit .env with your MIMIC data paths

# 2. Start core services (Kafka, Spark, MongoDB, Neo4j). Elasticsearch is optional:
make up
# Full stack with Elasticsearch (may require: sudo sysctl -w vm.max_map_count=262144):
# make up-all

# 3. (If no MIMIC-III access) Download MIMIC-IV demo data
python src/ingest/mimic_demo_setup.py

# 4. Run pipeline
make ingest-batch      # CSV → Parquet
# Start Kafka producer in separate terminal:
python src/ingest/kafka_producer.py --limit 50000
make ingest-stream     # Kafka → MongoDB

# 5. NLP concept extraction
spark-submit src/nlp/concept_extractor.py

# 6. Entity Resolution
make run-er

# 7. Build Knowledge Graph
# Apply schema first:
make neo4j-schema
# Then load data:
make build-graph

# 8. Run queries & benchmarks
python src/queries/queries.py
```

## Troubleshooting Docker Compose

**`Error response from daemon: No such container: …`** — Compose still references a container ID that no longer exists on the Docker daemon (stale state). Run:

```bash
make compose-reset   # docker compose down --remove-orphans
make up
```

Set **`COMPOSE_PROJECT_NAME=mimickg`** in `.env` (see `.env.example`) so the project name stays stable. If the error persists, remove named containers manually, then `make up`:

```bash
docker rm -f zookeeper kafka spark-master spark-worker mongodb neo4j 2>/dev/null || true
```

## Key Results

Run `jupyter notebook notebooks/` to explore:
- `nlp_comparison.ipynb` — scispaCy vs ClinicalBERT vs regex
- `er_analysis.ipynb` — Entity resolution precision/recall
- `graph_analytics.ipynb` — Community detection, PageRank
- `scalability.ipynb` — Throughput vs data size

## Project Structure

```
src/
├── ingest/          # Batch (Spark) + Streaming (Kafka) ingestion
├── nlp/             # Clinical NLP concept extraction
├── er/              # Entity resolution (blocking, features, classifier)
├── graph/           # Neo4j schema, graph builder, GDS analytics
├── queries/         # Cypher queries + benchmarks
└── benchmarks/      # Optimization experiment scripts
notebooks/           # Analysis & visualization
results/             # Benchmark CSVs (gitignored)
docs/                # Architecture diagram, ontology, design doc
```

## Data

This project uses [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) (requires PhysioNet credentialed access)  
or [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/) (public, no registration needed).

Place CSV files in the path configured by `MIMIC_DATA_DIR` in `.env`.
