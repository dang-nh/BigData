"""
Phase 6.2 — Graph Data Science analytics via Neo4j GDS library.

Algorithms:
  1. Louvain Community Detection on diagnosis co-occurrence graph
  2. PageRank on medication prescription graph
  3. Write community/rank back as node properties

Also benchmarks: GDS (in-memory projection) vs plain Cypher for community detection.

Usage:
  python src/graph/gds_analytics.py
"""

import os
import sys
import time
import csv
from pathlib import Path
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kg_paths

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
RESULTS_DIR = str(kg_paths.results_dir())


class GDSAnalytics:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    def close(self):
        self.driver.close()

    def run(self, query: str, **params):
        with self.driver.session() as s:
            return list(s.run(query, **params))

    # ── Step 0: Build co-occurrence edges in graph ─────────────────────────────

    COOCCURRENCE_QUERY = """
    MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d1:Diagnosis),
          (a)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
    WHERE d1.icd9_code < d2.icd9_code
    WITH d1, d2, count(a) AS co_count
    WHERE co_count >= $min_support
    MERGE (d1)-[r:CO_OCCURS_WITH]-(d2)
    SET r.count = co_count,
        r.pmi = log(toFloat(co_count))
    RETURN count(r) AS edges_created
    """

    def build_cooccurrence(self, min_support: int = 5):
        print(f"Building diagnosis co-occurrence edges (min_support={min_support}) ...")
        result = self.run(self.COOCCURRENCE_QUERY, min_support=min_support)
        edges = result[0]["edges_created"] if result else 0
        print(f"  Created {edges:,} CO_OCCURS_WITH edges")
        return edges

    # ── Step 1: GDS graph projection ──────────────────────────────────────────

    def drop_projection(self, name: str):
        try:
            self.run(f"CALL gds.graph.drop('{name}', false)")
        except Exception:
            pass

    def project_diagnosis_graph(self, proj_name: str = "diagnosis-cooccurrence"):
        self.drop_projection(proj_name)
        query = """
        CALL gds.graph.project(
          $proj_name,
          'Diagnosis',
          {
            CO_OCCURS_WITH: {
              orientation: 'UNDIRECTED',
              properties: ['count']
            }
          }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        result = self.run(query, proj_name=proj_name)
        if result:
            r = result[0]
            print(f"  Projected '{r['graphName']}': {r['nodeCount']:,} nodes, {r['relationshipCount']:,} rels")
        return proj_name

    def project_medication_graph(self, proj_name: str = "medication-prescription"):
        self.drop_projection(proj_name)
        query = """
        CALL gds.graph.project.cypher(
          $proj_name,
          'MATCH (n) WHERE n:Medication OR n:Admission RETURN id(n) AS id',
          'MATCH (a:Admission)-[:PRESCRIBED]->(m:Medication)
           RETURN id(a) AS source, id(m) AS target, 1.0 AS weight'
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        result = self.run(query, proj_name=proj_name)
        if result:
            r = result[0]
            print(f"  Projected '{r['graphName']}': {r['nodeCount']:,} nodes, {r['relationshipCount']:,} rels")
        return proj_name

    # ── Step 2: Louvain Community Detection ───────────────────────────────────

    def run_louvain(self, proj_name: str = "diagnosis-cooccurrence") -> dict:
        print("\nRunning Louvain community detection ...")
        t0 = time.perf_counter()
        query = """
        CALL gds.louvain.write(
          $proj_name,
          {
            writeProperty: 'community_id',
            relationshipWeightProperty: 'count',
            maxLevels: 10,
            maxIterations: 10,
            tolerance: 0.0001
          }
        )
        YIELD communityCount, modularity, ranLevels, nodePropertiesWritten
        RETURN communityCount, modularity, ranLevels, nodePropertiesWritten
        """
        result = self.run(query, proj_name=proj_name)
        elapsed = time.perf_counter() - t0
        if result:
            r = result[0]
            print(f"  Communities: {r['communityCount']}, Modularity: {r['modularity']:.4f}, "
                  f"Levels: {r['ranLevels']}, Time: {elapsed*1000:.0f}ms")
            return {
                "method": "gds_louvain",
                "communities": r["communityCount"],
                "modularity": r["modularity"],
                "time_ms": round(elapsed * 1000, 1),
            }
        return {"method": "gds_louvain", "communities": 0, "modularity": 0, "time_ms": 0}

    # ── Step 3: PageRank on medications ───────────────────────────────────────

    def run_pagerank(self, proj_name: str = "medication-prescription") -> dict:
        print("\nRunning PageRank on medication graph ...")
        t0 = time.perf_counter()
        query = """
        CALL gds.pageRank.write(
          $proj_name,
          {
            writeProperty: 'pagerank',
            maxIterations: 20,
            dampingFactor: 0.85,
            tolerance: 0.0000001
          }
        )
        YIELD ranIterations, didConverge, nodePropertiesWritten
        RETURN ranIterations, didConverge, nodePropertiesWritten
        """
        result = self.run(query, proj_name=proj_name)
        elapsed = time.perf_counter() - t0
        if result:
            r = result[0]
            print(f"  Iterations: {r['ranIterations']}, Converged: {r['didConverge']}, "
                  f"Time: {elapsed*1000:.0f}ms")
        return {"time_ms": round(elapsed * 1000, 1)}

    def top_pagerank_medications(self, top_k: int = 20) -> list[dict]:
        query = """
        MATCH (m:Medication)
        WHERE m.pagerank IS NOT NULL
        RETURN m.ndc AS ndc, m.drug AS drug, m.pagerank AS rank
        ORDER BY rank DESC LIMIT $k
        """
        return [dict(r) for r in self.run(query, k=top_k)]

    def top_communities(self, top_k: int = 10) -> list[dict]:
        query = """
        MATCH (d:Diagnosis)
        WHERE d.community_id IS NOT NULL
        WITH d.community_id AS cid, collect(d.short_title)[..5] AS titles, count(*) AS size
        ORDER BY size DESC LIMIT $k
        RETURN cid, size, titles
        """
        return [dict(r) for r in self.run(query, k=top_k)]

    # ── Step 4: Baseline — pure Cypher "community" (label propagation sim) ────

    def cypher_label_propagation(self) -> dict:
        """Naive Cypher traversal to count weakly connected components."""
        print("\nCypher baseline: weakly connected components ...")
        t0 = time.perf_counter()
        query = """
        MATCH (d:Diagnosis)-[:CO_OCCURS_WITH]-()
        WITH DISTINCT d
        RETURN count(d) AS connected_nodes
        """
        result = self.run(query)
        elapsed = time.perf_counter() - t0
        nodes = result[0]["connected_nodes"] if result else 0
        print(f"  Connected nodes: {nodes:,}, Time: {elapsed*1000:.0f}ms")
        return {"method": "cypher_wcc", "time_ms": round(elapsed * 1000, 1), "nodes": nodes}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gds = GDSAnalytics()

    # Build co-occurrence edges
    gds.build_cooccurrence(min_support=5)

    # Community detection
    diag_proj = gds.project_diagnosis_graph()
    louvain_result = gds.run_louvain(diag_proj)
    cypher_result = gds.cypher_label_propagation()

    print("\nTop 10 diagnosis communities:")
    for c in gds.top_communities(10):
        print(f"  Community {c['cid']}: {c['size']} diagnoses — {', '.join(c['titles'])}")

    # PageRank
    med_proj = gds.project_medication_graph()
    pr_result = gds.run_pagerank(med_proj)

    print("\nTop 20 medications by PageRank:")
    for r in gds.top_pagerank_medications(20):
        print(f"  {r['drug']:40s}  rank={r['rank']:.6f}")

    # Benchmark comparison: GDS vs Cypher
    benchmark = [
        {"algorithm": "Louvain", "backend": "GDS", "time_ms": louvain_result["time_ms"],
         "communities": louvain_result.get("communities"), "modularity": louvain_result.get("modularity")},
        {"algorithm": "WCC-sim", "backend": "Cypher", "time_ms": cypher_result["time_ms"],
         "communities": None, "modularity": None},
        {"algorithm": "PageRank", "backend": "GDS", "time_ms": pr_result["time_ms"],
         "communities": None, "modularity": None},
    ]

    out_path = f"{RESULTS_DIR}/gds_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "backend", "time_ms", "communities", "modularity"])
        writer.writeheader()
        writer.writerows(benchmark)
    print(f"\nGDS benchmark saved to {out_path}")

    gds.close()


if __name__ == "__main__":
    main()
