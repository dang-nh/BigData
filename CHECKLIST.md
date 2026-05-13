# Knowledge Graph Integration — Project Checklist

**Theme**: Medical Diagnosis Support  
**Data**: MIMIC-III (PhysioNet)  
**Stack**: Kafka · Spark · MongoDB · Neo4j · scispaCy

---

## Phase 1 — Setup & Design (Tuần 1–2)

### 1.1 Infrastructure
- [x] Viết `docker-compose.yml` với đầy đủ services:
  - [x] Kafka + Zookeeper
  - [x] Spark (master + worker)
  - [x] Neo4j Community
  - [x] MongoDB
  - [x] Elasticsearch (optional)
- [ ] Test toàn bộ services khởi động không lỗi *(cần chạy thực tế)*
- [x] Viết `Makefile` với các targets: `up`, `down`, `reset`, `logs`

### 1.2 Repository Structure
- [ ] Khởi tạo Git repo *(chạy `git init` + `git add . && git commit`)*
- [x] Tạo cấu trúc thư mục (xem mục "Repo Layout" bên dưới)
- [x] Viết `requirements.txt` / `pyproject.toml`
- [x] Viết `README.md` tổng quan

### 1.3 Design Documents
- [ ] Vẽ Architecture Diagram (`docs/architecture.png`) *(dùng draw.io hoặc Excalidraw)*
- [ ] Vẽ Ontology Diagram (`docs/ontology.png`) *(dùng Protégé hoặc draw.io)*
- [x] Viết Problem Statement + 3 Use Cases cụ thể (`docs/design.md`)
- [ ] Xin access MIMIC-III tại PhysioNet (làm ngay — cần CITI training)
  - [x] Fallback: `src/ingest/mimic_demo_setup.py` tự động tải MIMIC-IV demo

---

## Phase 2 — Data Ingestion (Tuần 3)

### 2.1 Batch Ingest (MIMIC-III CSV → Parquet)
- [x] Viết `src/ingest/batch_loader.py`
  - [x] Đọc các bảng: PATIENTS, ADMISSIONS, DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS, LABEVENTS, NOTEEVENTS, CAREGIVERS
  - [x] Validate schema, log null counts
  - [x] Ghi ra Parquet phân partition theo `subject_id`
- [ ] Benchmark throughput (records/s) + ghi vào `results/ingestion_benchmark.csv` *(chạy thực tế)*

### 2.2 Streaming Sim (NOTEEVENTS → Kafka → MongoDB)
- [x] Viết `src/ingest/kafka_producer.py` — đọc NOTEEVENTS, push theo `charttime`
- [x] Viết `src/ingest/kafka_consumer.py` — Spark Structured Streaming ghi vào MongoDB
- [ ] Test end-to-end latency Kafka → MongoDB *(chạy thực tế)*
- [ ] Benchmark throughput streaming pipeline *(chạy thực tế)*

---

## Phase 3 — NLP & Entity Extraction (Tuần 4)

### 3.1 Concept Extraction
- [x] Viết `src/nlp/concept_extractor.py`
  - [x] Tích hợp **scispaCy** (`en_core_sci_md`) + UMLS entity linker
  - [x] Dùng `mapPartitions` trong Spark UDF để load model 1 lần/partition
  - [x] Output: `(note_id, cui, name, score, char_start, char_end)`
- [ ] Viết unit tests cho extractor *(thêm vào `tests/`)*

### 3.2 NLP Comparison Experiment *(Optimization #3)*
- [x] Baseline: regex/keyword matching từ ICD titles (`src/nlp/nlp_comparison.py`)
- [x] Method A: scispaCy `en_core_sci_md`
- [x] Method B: ClinicalBERT trên subset 500–10k notes
- [ ] Ghi kết quả vào `results/nlp_comparison.csv` *(chạy thực tế)*
- [x] Vẽ biểu đồ so sánh (`notebooks/nlp_comparison.ipynb`)

---

## Phase 4 — Entity Resolution (Tuần 5) ⭐

### 4.1 Candidate Generation
- [x] Viết `src/er/blocking.py`
  - [x] Blocking cho Drug: theo `ndc_prefix`, `first_word`
  - [x] Blocking cho Diagnosis: theo `icd_chapter`
- [x] Implement **MinHash LSH** trên Spark MLlib
- [x] So sánh brute-force vs LSH: `src/er/lsh_sweep.py` *(Optimization #2)*

### 4.2 Similarity Features
- [x] Viết `src/er/features.py`
  - [x] Jaro-Winkler distance (tên drug)
  - [x] Levenshtein distance (normalized)
  - [x] Token overlap (Jaccard on word sets)
  - [x] Exact code match (NDC prefix, ICD category)

### 4.3 ER Classifier
- [x] Auto gold set từ heuristics (same NDC product, high JW similarity)
- [x] Train RandomForest classifier (Spark MLlib) — `src/er/classifier.py`
- [x] Evaluate: Precision / Recall / F1 / AUC
- [x] Output: `same_as_edges.parquet`
- [ ] Ghi kết quả vào `results/er_evaluation.csv` *(chạy thực tế)*

---

## Phase 5 — Build Knowledge Graph (Tuần 6)

### 5.1 Schema & Constraints
- [x] Viết `src/graph/schema.cypher` — tạo constraints + indexes
  - [x] `UNIQUE` trên `:Patient(subject_id)`, `:Admission(hadm_id)`, `:Diagnosis(icd9_code)`, `:Medication(ndc)`, `:Concept(cui)`
  - [x] Index trên `:ClinicalNote(row_id)`, thêm 6 indexes khác

### 5.2 Graph Loading
- [x] Viết `src/graph/graph_builder.py` dùng `neo4j-spark-connector`
  - [x] Load nodes theo thứ tự: Patient → Admission → Diagnosis → Medication → LabTest → Concept
  - [x] Load relationships: HAS_ADMISSION, HAS_DIAGNOSIS, PRESCRIBED, MENTIONS
  - [x] Load `same_as` edges từ ER output
- [x] Benchmark batch size script: `src/benchmarks/neo4j_batch_benchmark.py` *(Optimization #4)*
- [ ] Ghi kết quả vào `results/graph_load_benchmark.csv` *(chạy thực tế)*

### 5.3 Kiểm tra graph
- [x] Assert số nodes/edges match expected counts (`src/graph/graph_verify.py`)
- [x] Sample 10 patients, verify subgraph đúng

---

## Phase 6 — Graph Queries & Analytics (Tuần 7)

### 6.1 Cypher Queries
- [x] Viết `src/queries/queries.py` với 5 queries:
  - [x] **Q1**: Shortest path giữa 2 bệnh nhân qua shared diagnosis
  - [x] **Q2**: Top-k diagnoses liên quan cho một symptom set (differential diagnosis)
  - [x] **Q3**: Cohort discovery — bệnh nhân có admission trajectory tương tự
  - [x] **Q4**: Medication–Diagnosis co-occurrence
  - [x] **Q5**: Most prescribed drug per diagnosis

### 6.2 Graph Analytics (Neo4j GDS)
- [x] Tạo GDS graph projection (`src/graph/gds_analytics.py`)
- [x] Chạy **Louvain Community Detection** trên diagnosis co-occurrence
- [x] Chạy **PageRank** trên medication graph
- [x] Export community labels + pagerank về Neo4j properties

### 6.3 Query Benchmark *(Optimization #6)*
- [x] Benchmark Q1–Q5: latency p50/p95/p99 (50 runs) — `src/queries/queries.py`
- [ ] Chạy và ghi vào `results/query_benchmark.csv` *(chạy thực tế)*
- [x] So sánh GDS vs thuần Cypher (`src/graph/gds_analytics.py` — cypher_label_propagation)

---

## Phase 7 — Evaluation & Optimization (Tuần 8)

### 7.1 Optimization Experiments (cần có biểu đồ cho từng cái)

| # | Thí nghiệm | Script | Kết quả |
|---|---|---|---|
| O1 | Spark skew handling: salting vs không | `src/benchmarks/spark_skew.py` | `results/spark_skew.csv` |
| O2 | ER: brute-force vs LSH (vary b,r) | `src/er/lsh_sweep.py` | `results/er_lsh_sweep.csv` |
| O3 | NLP: scispaCy vs ClinicalBERT vs regex | `src/nlp/nlp_comparison.py` | `results/nlp_comparison.csv` |
| O4 | Neo4j write batch size | `src/benchmarks/neo4j_batch_benchmark.py` | `results/graph_load_benchmark.csv` |
| O5 | Spark partition strategy | `src/benchmarks/partition_benchmark.py` | `results/spark_partition.csv` |
| O6 | GDS vs Cypher; query latency | `src/queries/queries.py` + `gds_analytics.py` | `results/query_benchmark.csv` |

> **Tất cả scripts đã viết xong** — chỉ cần chạy khi có data và điền số vào notebooks.

### 7.2 Scalability Test
- [x] Script `src/benchmarks/scalability.py`: 10%, 50%, 100% data
- [x] Vẽ scalability chart (`notebooks/scalability.ipynb`)
- [ ] Điền số thực tế sau khi chạy

### 7.3 Case Studies
- [ ] Chọn 3 bệnh nhân thú vị (vd: heart failure, sepsis, diabetes)
- [ ] Capture Neo4j Bloom screenshots (sau khi graph loaded)
- [ ] Viết narrative insight cho mỗi case (100–150 từ)

---

## Phase 8 — Report & Presentation

### 8.1 Báo cáo (IEEE format)
- [ ] Abstract
- [ ] Introduction + Contributions
- [ ] Related Work (≥ 15 refs)
- [ ] System Design (architecture, ontology)
- [ ] Implementation Details
- [ ] Entity Resolution Section
- [ ] Evaluation (all experiments + charts)
- [ ] Discussion (limitations, future work)
- [ ] Conclusion

### 8.2 Slides & Demo
- [ ] Slide deck (15 slides)
- [ ] Demo video 3–5 phút (record Neo4j Bloom + query results)
- [ ] Chuẩn bị Q&A: lý do chọn Neo4j thay vì JanusGraph, MinHash LSH params, ER precision/recall tradeoffs

---

## Repo Layout

```
final-proj/
├── docker/
│   ├── docker-compose.yml
│   └── spark/Dockerfile
├── src/
│   ├── ingest/
│   │   ├── batch_loader.py
│   │   ├── kafka_producer.py
│   │   └── kafka_consumer.py
│   ├── nlp/
│   │   └── concept_extractor.py
│   ├── er/
│   │   ├── blocking.py
│   │   ├── features.py
│   │   └── classifier.py
│   ├── graph/
│   │   ├── schema.cypher
│   │   └── graph_builder.py
│   └── queries/
│       └── queries.py
├── notebooks/
│   ├── nlp_comparison.ipynb
│   ├── er_analysis.ipynb
│   ├── graph_analytics.ipynb
│   └── scalability.ipynb
├── results/          # CSV benchmark outputs
├── docs/
│   ├── design.md
│   ├── architecture.png
│   └── ontology.png
├── tests/
├── data/             # gitignored, local only
├── report/
├── CHECKLIST.md
├── Makefile
├── requirements.txt
└── README.md
```

---

## Progress Tracker

| Phase | Status | Notes |
|---|---|---|
| 1 - Setup & Design | 🔄 In progress | Code done. **TODO**: git init, vẽ diagrams bằng draw.io, xin MIMIC-III access |
| 2 - Data Ingestion | 🔄 In progress | Code done. **TODO**: chạy thực tế với data, ghi benchmark |
| 3 - NLP Extraction | 🔄 In progress | Code done. **TODO**: chạy nlp_comparison.py, điền F1 nếu có gold |
| 4 - Entity Resolution | 🔄 In progress | Code done. **TODO**: chạy pipeline, lấy số thực tế |
| 5 - Graph Build | 🔄 In progress | Code done. **TODO**: chạy graph_builder, verify, benchmark batch size |
| 6 - Queries & Analytics | 🔄 In progress | Code done. **TODO**: chạy GDS, query benchmark với Neo4j thực |
| 7 - Evaluation | 🔄 In progress | Scripts done. **TODO**: chạy tất cả, điền số vào notebooks, case studies |
| 8 - Report | ⬜ Not started | Có thể bắt đầu sau khi có một số benchmark results |

> Update status: ⬜ Not started → 🔄 In progress → ✅ Done
