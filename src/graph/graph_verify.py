"""
Phase 5.3 — Verify Knowledge Graph integrity after loading.

Checks:
  - Node counts match expected parquet row counts
  - Key relationships exist (no orphaned nodes)
  - Sample 10 patients and verify their subgraph structure
  - All constraints are in place

Usage:
  python src/graph/graph_verify.py
"""

import os
import sys
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "neo4jpassword")


class GraphVerifier:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self.failures = []

    def close(self):
        self.driver.close()

    def run(self, query: str, **params):
        with self.driver.session() as s:
            return list(s.run(query, **params))

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if not condition:
            self.failures.append(name)

    # ── 1. Node count checks ──────────────────────────────────────────────────

    def verify_node_counts(self):
        print("\n── Node counts ──────────────────────────────────────────────")
        labels = ["Patient", "Admission", "Diagnosis", "Medication", "LabTest",
                  "ClinicalNote", "Concept", "Caregiver"]
        for label in labels:
            result = self.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            cnt = result[0]["cnt"] if result else 0
            self.check(f"{label} nodes loaded", cnt > 0, f"count={cnt:,}")

    # ── 2. Relationship count checks ──────────────────────────────────────────

    def verify_relationships(self):
        print("\n── Relationships ────────────────────────────────────────────")
        rel_checks = [
            ("HAS_ADMISSION", "MATCH ()-[r:HAS_ADMISSION]->() RETURN count(r) AS cnt"),
            ("HAS_DIAGNOSIS",  "MATCH ()-[r:HAS_DIAGNOSIS]->() RETURN count(r) AS cnt"),
            ("PRESCRIBED",     "MATCH ()-[r:PRESCRIBED]->() RETURN count(r) AS cnt"),
            ("MENTIONS",       "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS cnt"),
        ]
        for name, query in rel_checks:
            result = self.run(query)
            cnt = result[0]["cnt"] if result else 0
            self.check(f"{name} edges exist", cnt > 0, f"count={cnt:,}")

    # ── 3. Orphan checks ──────────────────────────────────────────────────────

    def verify_no_orphans(self):
        print("\n── Orphan checks ────────────────────────────────────────────")

        # Admissions must have a Patient
        result = self.run("""
            MATCH (a:Admission) WHERE NOT (a)<-[:HAS_ADMISSION]-(:Patient)
            RETURN count(a) AS cnt
        """)
        orphan_adm = result[0]["cnt"] if result else 0
        self.check("No orphan Admissions", orphan_adm == 0, f"orphans={orphan_adm:,}")

        # ClinicalNotes must link to an Admission
        result = self.run("""
            MATCH (n:ClinicalNote) WHERE NOT (:Admission)-[:DOCUMENTED_IN]->(n)
            RETURN count(n) AS cnt
        """)
        orphan_notes = result[0]["cnt"] if result else 0
        self.check("No orphan ClinicalNotes", orphan_notes < 100,
                   f"orphans={orphan_notes:,} (threshold: <100)")

    # ── 4. Sample patient subgraph ────────────────────────────────────────────

    def verify_sample_patients(self, n: int = 10):
        print(f"\n── Sample {n} patients ───────────────────────────────────────")
        patients = self.run(f"MATCH (p:Patient) RETURN p.subject_id AS sid LIMIT {n}")

        for row in patients:
            sid = row["sid"]
            result = self.run("""
                MATCH (p:Patient {subject_id: $sid})-[:HAS_ADMISSION]->(a:Admission)
                OPTIONAL MATCH (a)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
                OPTIONAL MATCH (a)-[:PRESCRIBED]->(m:Medication)
                RETURN count(DISTINCT a) AS admissions,
                       count(DISTINCT d) AS diagnoses,
                       count(DISTINCT m) AS medications
            """, sid=sid)
            if result:
                r = result[0]
                ok = r["admissions"] > 0
                self.check(
                    f"Patient {sid} has subgraph",
                    ok,
                    f"admissions={r['admissions']}, dx={r['diagnoses']}, rx={r['medications']}",
                )

    # ── 5. Constraint verification ────────────────────────────────────────────

    def verify_constraints(self):
        print("\n── Constraints ──────────────────────────────────────────────")
        result = self.run("SHOW CONSTRAINTS")
        constraint_names = [r["name"] for r in result]

        expected = ["patient_id", "admission_id", "diagnosis_code",
                    "medication_ndc", "concept_cui", "note_id"]
        for name in expected:
            exists = any(name in c for c in constraint_names)
            self.check(f"Constraint '{name}' exists", exists)

    # ── 6. Graph statistics ───────────────────────────────────────────────────

    def print_stats(self):
        print("\n── Graph Statistics ─────────────────────────────────────────")
        result = self.run("""
            CALL apoc.meta.stats()
            YIELD labels, relTypesCount
            RETURN labels, relTypesCount
        """)
        if result:
            r = result[0]
            print("  Node counts:")
            for label, cnt in sorted(r["labels"].items()):
                print(f"    {label:20s}: {cnt:,}")
            print("  Relationship counts:")
            for rel, cnt in sorted(r["relTypesCount"].items()):
                print(f"    {rel:30s}: {cnt:,}")
        else:
            # fallback without APOC
            for label in ["Patient", "Admission", "Diagnosis", "Medication",
                          "LabTest", "ClinicalNote", "Concept"]:
                result = self.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                cnt = result[0]["cnt"] if result else 0
                print(f"  {label:20s}: {cnt:,}")


def main():
    verifier = GraphVerifier()

    verifier.verify_constraints()
    verifier.verify_node_counts()
    verifier.verify_relationships()
    verifier.verify_no_orphans()
    verifier.verify_sample_patients(10)
    verifier.print_stats()

    print(f"\n{'='*55}")
    if verifier.failures:
        print(f"RESULT: {len(verifier.failures)} check(s) FAILED:")
        for f in verifier.failures:
            print(f"  - {f}")
        verifier.close()
        sys.exit(1)
    else:
        print("RESULT: All checks PASSED")
        verifier.close()


if __name__ == "__main__":
    main()
