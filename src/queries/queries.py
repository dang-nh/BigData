"""
Knowledge Graph queries: 5 clinical use cases + benchmarking.
"""

import os
import time
import statistics
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")


class KGQueries:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    def close(self):
        self.driver.close()

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return list(session.run(query, **params))

    # ── Q1: Shortest path between two patients via shared diagnoses ───────────

    Q1 = """
    MATCH (p1:Patient {subject_id: $subject_id_a}),
          (p2:Patient {subject_id: $subject_id_b})
    MATCH path = shortestPath(
      (p1)-[:HAS_ADMISSION*..2]-(:Admission)-[:HAS_DIAGNOSIS*..2]-(p2)
    )
    RETURN path, length(path) AS path_length
    LIMIT 1
    """

    def q1_patient_connection(self, subject_id_a: str, subject_id_b: str):
        return self.run(self.Q1, subject_id_a=subject_id_a, subject_id_b=subject_id_b)

    # ── Q2: Differential diagnosis — top diagnoses for a set of concepts ─────

    Q2 = """
    MATCH (c:Concept) WHERE c.name IN $concept_names
    MATCH (c)<-[:MENTIONS]-(note:ClinicalNote)<-[:DOCUMENTED_IN]-(adm:Admission)
    MATCH (adm)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
    WITH d, count(DISTINCT adm) AS support
    ORDER BY support DESC
    RETURN d.icd9_code AS code, d.short_title AS title, support
    LIMIT $top_k
    """

    def q2_differential_diagnosis(self, concept_names: list[str], top_k: int = 10):
        return self.run(self.Q2, concept_names=concept_names, top_k=top_k)

    # ── Q3: Cohort discovery — admissions with similar diagnosis patterns ─────

    Q3 = """
    MATCH (ref:Admission {hadm_id: $hadm_id})-[:HAS_DIAGNOSIS]->(d:Diagnosis)
    WITH collect(d.icd9_code) AS ref_codes
    MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
    WHERE a.hadm_id <> $hadm_id
    WITH a, collect(d2.icd9_code) AS a_codes, ref_codes
    WITH a, a_codes, ref_codes,
         size([x IN ref_codes WHERE x IN a_codes]) AS intersection,
         size(ref_codes + [x IN a_codes WHERE NOT x IN ref_codes]) AS union_size
    WITH a, toFloat(intersection) / union_size AS jaccard
    WHERE jaccard > $threshold
    RETURN a.hadm_id AS hadm_id, jaccard
    ORDER BY jaccard DESC
    LIMIT $top_k
    """

    def q3_similar_cohort(self, hadm_id: str, threshold: float = 0.4, top_k: int = 20):
        return self.run(self.Q3, hadm_id=hadm_id, threshold=threshold, top_k=top_k)

    # ── Q4: Medication–Diagnosis co-occurrence ────────────────────────────────

    Q4 = """
    MATCH (m:Medication {ndc: $ndc})<-[:PRESCRIBED]-(a:Admission)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
    WITH d, count(a) AS co_count
    ORDER BY co_count DESC
    RETURN d.icd9_code AS code, d.short_title AS title, co_count
    LIMIT $top_k
    """

    def q4_medication_diagnoses(self, ndc: str, top_k: int = 15):
        return self.run(self.Q4, ndc=ndc, top_k=top_k)

    # ── Q5: Most prescribed medication per diagnosis ──────────────────────────

    Q5 = """
    MATCH (d:Diagnosis {icd9_code: $icd9_code})<-[:HAS_DIAGNOSIS]-(a:Admission)-[:PRESCRIBED]->(m:Medication)
    WITH m, count(a) AS rx_count
    ORDER BY rx_count DESC
    RETURN m.ndc AS ndc, m.drug AS drug, rx_count
    LIMIT $top_k
    """

    def q5_top_drugs_for_diagnosis(self, icd9_code: str, top_k: int = 10):
        return self.run(self.Q5, icd9_code=icd9_code, top_k=top_k)


def benchmark_query(kg: KGQueries, name: str, fn, n_runs: int = 50) -> dict:
    """Run a query n_runs times and return latency stats in ms."""
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    print(f"  {name}: p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    return {"query": name, "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "p99_ms": round(p99, 2)}


def run_benchmarks():
    kg = KGQueries()

    # Pick real IDs from your data
    PATIENT_A = os.getenv("BENCH_PATIENT_A", "10006")
    PATIENT_B = os.getenv("BENCH_PATIENT_B", "10011")
    HADM_ID = os.getenv("BENCH_HADM_ID", "150750")
    NDC = os.getenv("BENCH_NDC", "00074187115")
    ICD_CODE = os.getenv("BENCH_ICD_CODE", "41401")  # Coronary atherosclerosis

    results = []
    print("\n=== Query Benchmark ===")
    results.append(benchmark_query(kg, "Q1-shortest-path",
        lambda: kg.q1_patient_connection(PATIENT_A, PATIENT_B)))
    results.append(benchmark_query(kg, "Q2-differential-dx",
        lambda: kg.q2_differential_diagnosis(["chest pain", "dyspnea"], top_k=10)))
    results.append(benchmark_query(kg, "Q3-cohort-discovery",
        lambda: kg.q3_similar_cohort(HADM_ID)))
    results.append(benchmark_query(kg, "Q4-drug-dx-cooccurrence",
        lambda: kg.q4_medication_diagnoses(NDC)))
    results.append(benchmark_query(kg, "Q5-top-drugs-for-dx",
        lambda: kg.q5_top_drugs_for_diagnosis(ICD_CODE)))

    import csv, os as _os
    _os.makedirs("/data/results", exist_ok=True)
    with open("/data/results/query_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "p50_ms", "p95_ms", "p99_ms"])
        writer.writeheader()
        writer.writerows(results)
    print("Saved to /data/results/query_benchmark.csv")
    kg.close()


if __name__ == "__main__":
    run_benchmarks()
