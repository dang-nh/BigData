// ── Constraints (enforce uniqueness & create implicit index) ──────────────────

CREATE CONSTRAINT patient_id IF NOT EXISTS
  FOR (p:Patient) REQUIRE p.subject_id IS UNIQUE;

CREATE CONSTRAINT admission_id IF NOT EXISTS
  FOR (a:Admission) REQUIRE a.hadm_id IS UNIQUE;

CREATE CONSTRAINT diagnosis_code IF NOT EXISTS
  FOR (d:Diagnosis) REQUIRE d.icd9_code IS UNIQUE;

CREATE CONSTRAINT procedure_code IF NOT EXISTS
  FOR (pr:Procedure) REQUIRE pr.icd9_code IS UNIQUE;

CREATE CONSTRAINT medication_ndc IF NOT EXISTS
  FOR (m:Medication) REQUIRE m.ndc IS UNIQUE;

CREATE CONSTRAINT labtest_id IF NOT EXISTS
  FOR (l:LabTest) REQUIRE l.itemid IS UNIQUE;

CREATE CONSTRAINT note_id IF NOT EXISTS
  FOR (n:ClinicalNote) REQUIRE n.row_id IS UNIQUE;

CREATE CONSTRAINT concept_cui IF NOT EXISTS
  FOR (c:Concept) REQUIRE c.cui IS UNIQUE;

CREATE CONSTRAINT caregiver_id IF NOT EXISTS
  FOR (cg:Caregiver) REQUIRE cg.caregiver_id IS UNIQUE;

// ── Additional indexes for frequent query patterns ────────────────────────────

CREATE INDEX patient_gender IF NOT EXISTS FOR (p:Patient) ON (p.gender);
CREATE INDEX admission_type IF NOT EXISTS FOR (a:Admission) ON (a.admission_type);
CREATE INDEX diagnosis_title IF NOT EXISTS FOR (d:Diagnosis) ON (d.short_title);
CREATE INDEX medication_drug IF NOT EXISTS FOR (m:Medication) ON (m.drug);
CREATE INDEX note_category IF NOT EXISTS FOR (n:ClinicalNote) ON (n.category);
CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);
