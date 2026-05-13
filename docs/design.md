# System Design — Medical Knowledge Graph

## Problem Statement

Electronic Health Records (EHRs) store critical clinical information across disconnected silos: structured tables (diagnoses, medications, lab results) and unstructured free-text (clinical notes). Clinicians and researchers cannot easily discover hidden relationships across these sources — for example, which medications co-occur with a rare diagnosis, or which patients share a similar disease trajectory.

This project builds a **unified Knowledge Graph** over MIMIC-III that:
1. Integrates structured and unstructured clinical data
2. Resolves entity ambiguities (drug aliases, ICD code overlaps) via Entity Resolution
3. Enables complex graph traversal queries for clinical decision support

## Use Cases

### UC-1: Differential Diagnosis Assistance
A clinician documents symptoms in a clinical note. The system extracts medical concepts (scispaCy + UMLS), traverses the graph to find historically co-occurring diagnoses, and returns a ranked list of differential diagnoses weighted by support and recency.

**Query pattern**: `Concept → ClinicalNote → Admission → Diagnosis`

### UC-2: Cohort Discovery
A researcher studying sepsis patients wants to find all admissions with a similar diagnosis/procedure trajectory to a reference case. The system computes pairwise Jaccard similarity over diagnosis sets and returns the closest cohort.

**Query pattern**: Pattern-matching on `(Admission)-[:HAS_DIAGNOSIS*]-(Diagnosis)` with set similarity

### UC-3: Pharmacological Network Analysis
A pharmacologist wants to understand which drugs are most central in treatment pathways and which drug–diagnosis co-occurrences are statistically anomalous (high PMI). PageRank on the medication graph identifies key hubs; PMI computation flags unusual prescriptions.

**Query pattern**: GDS PageRank on `(Medication)-[:PRESCRIBED]-(Admission)-[:HAS_DIAGNOSIS]-(Diagnosis)`

## Architecture Justification

| Component | Chosen | Alternatives Considered | Reason |
|---|---|---|---|
| Graph DB | Neo4j 5 + GDS | JanusGraph, TigerGraph | Native graph storage, mature GDS library (Louvain, PageRank built-in), Cypher DSL |
| Stream broker | Kafka | Pulsar, Kinesis | Industry standard, Spark native integration |
| Batch engine | Spark 3.5 | Flink, Dask | Best-in-class for large-scale ETL + MLlib for ER |
| Document store | MongoDB | Cassandra, CouchDB | Schema-flexible for clinical notes, fast document writes |
| NLP | scispaCy | spaCy general, MetaMap | Biomedical pre-trained, UMLS linker, runs on CPU |
| ER blocking | MinHash LSH | Sorted-neighborhood, canopy | Scales to millions of pairs on Spark, tunable recall |

## Ontology

### Entity Types

| Label | Key Property | Description |
|---|---|---|
| `:Patient` | `subject_id` | De-identified patient |
| `:Admission` | `hadm_id` | Single hospital stay |
| `:Diagnosis` | `icd9_code` | ICD-9 diagnosis code |
| `:Procedure` | `icd9_code` | ICD-9 procedure code |
| `:Medication` | `ndc` | Drug (NDC code) |
| `:LabTest` | `itemid` | Lab item definition |
| `:ClinicalNote` | `row_id` | Free-text note |
| `:Concept` | `cui` | UMLS Concept Unique Identifier |
| `:Caregiver` | `caregiver_id` | Physician / nurse |

### Relationships

```
(Patient)       -[:HAS_ADMISSION]->          (Admission)
(Admission)     -[:HAS_DIAGNOSIS {seq_num}]-> (Diagnosis)
(Admission)     -[:UNDERWENT]->              (Procedure)
(Admission)     -[:PRESCRIBED {dose,route}]-> (Medication)
(Admission)     -[:HAD_LAB_RESULT {value}]->  (LabTest)
(Admission)     -[:DOCUMENTED_IN]->           (ClinicalNote)
(ClinicalNote)  -[:MENTIONS {score}]->        (Concept)
(Diagnosis)     -[:CO_OCCURS_WITH {pmi}]->    (Diagnosis)
(Medication)    -[:SAME_AS {confidence}]->    (Medication)   // ER output
(Diagnosis)     -[:SAME_AS {confidence}]->    (Diagnosis)    // ER output
```

## Data Flow

```
MIMIC-III
   │
   ├── Batch tables (CSV)
   │     └─► Spark ETL ─► Parquet ─► Spark NLP ─► concepts.parquet
   │                                     └─► ER pipeline ─► same_as.parquet
   │                                           └─► Neo4j graph builder
   │
   └── NOTEEVENTS (streaming sim)
         └─► Kafka producer ─► Kafka topic ─► Spark Structured Streaming ─► MongoDB
                                                        └─► Concept Extraction (async)
```

## Scalability Considerations

- Spark partitioning by `subject_id` ensures data locality for patient-centric joins
- MinHash LSH reduces O(n²) ER comparison to O(n log n) expected
- Neo4j GDS in-memory projection enables sub-second analytics on millions of edges
- MongoDB sharding on `subject_id` for horizontal scale of note store
