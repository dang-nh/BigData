"""
Optimization #4: Neo4j Python-driver batch size benchmark.
Tests batch sizes 500→50000 when loading Diagnosis nodes.
Outputs data/results/graph_load_benchmark.csv
"""
import os, time, csv
from neo4j import GraphDatabase

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
RESULTS    = os.getenv("RESULTS_DIR",    "/home/ntan/final-proj/data/results")

BATCH_SIZES = [500, 1000, 5000, 10000, 25000, 50000]
N_ROWS      = 10_000   # synthetic node count for benchmark

def make_rows(n):
    return [{"icd9_code": f"BENCH{i:05d}",
             "short_title": f"Benchmark Diagnosis {i}",
             "long_title":  f"Long title for benchmark node {i}"} for i in range(n)]

QUERY = """
UNWIND $rows AS row
MERGE (d:BenchDiagnosis {icd9_code: row.icd9_code})
  SET d.short_title = row.short_title,
      d.long_title  = row.long_title
"""

def cleanup(session):
    session.run("MATCH (n:BenchDiagnosis) DETACH DELETE n")

def run_batch(driver, rows, batch_size):
    t0 = time.time()
    with driver.session() as session:
        cleanup(session)
        for i in range(0, len(rows), batch_size):
            session.run(QUERY, rows=rows[i:i+batch_size])
    return time.time() - t0

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    rows   = make_rows(N_ROWS)
    os.makedirs(RESULTS, exist_ok=True)
    results = []

    print(f"Benchmarking {N_ROWS:,} rows across batch sizes: {BATCH_SIZES}")
    for bs in BATCH_SIZES:
        elapsed = run_batch(driver, rows, bs)
        rps     = N_ROWS / max(elapsed, 1e-6)
        print(f"  batch={bs:>6,}: {elapsed:.2f}s  {rps:,.0f} rec/s")
        results.append({"batch_size": bs, "rows": N_ROWS,
                        "time_s": round(elapsed, 3),
                        "records_per_s": round(rps, 0)})

    # cleanup
    with driver.session() as s:
        cleanup(s)
    driver.close()

    out = f"{RESULTS}/graph_load_benchmark.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_size","rows","time_s","records_per_s"])
        w.writeheader(); w.writerows(results)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
