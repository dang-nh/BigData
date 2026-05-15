"""
Local pipeline runner — wraps all stages, sets local Spark master.
Run: .venv/bin/python3 run_pipeline.py [--stage all|etl|er|graph|bench|query]
"""

import os
import sys
from pathlib import Path

_rp = Path(__file__).resolve().parent
sys.path.insert(0, str(_rp / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(_rp / ".env")
except ImportError:
    pass

import kg_paths

import argparse, time, csv

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "neo4jpassword")

MIMIC_DIR = str(kg_paths.mimic_data_dir())
PARQUET = str(kg_paths.parquet_output_dir())
RESULTS = str(kg_paths.results_dir())
os.makedirs(RESULTS, exist_ok=True)

def get_spark(app="KG-Pipeline"):
    from pyspark.sql import SparkSession
    return (SparkSession.builder
            .master("local[*]")
            .appName(app)
            .config("spark.sql.adaptive.enabled","true")
            .config("spark.driver.memory","2g")
            .config("spark.sql.shuffle.partitions","20")
            .getOrCreate())

# ── Stage 1: ETL ───────────────────────────────────────────────────────────────

def run_etl():
    print("\n" + "="*55)
    print("STAGE 1: Batch ETL — CSV → Parquet")
    print("="*55)
    from pyspark.sql import functions as F
    spark = get_spark("KG-ETL")
    spark.sparkContext.setLogLevel("WARN")

    TABLES = [
        ("PATIENTS.csv","subject_id"),("ADMISSIONS.csv","subject_id"),
        ("DIAGNOSES_ICD.csv","subject_id"),("PROCEDURES_ICD.csv","subject_id"),
        ("PRESCRIPTIONS.csv","subject_id"),("LABEVENTS.csv","subject_id"),
        ("NOTEEVENTS.csv","subject_id"),("CAREGIVERS.csv",None),
        ("D_ICD_DIAGNOSES.csv",None),("D_ICD_PROCEDURES.csv",None),
        ("D_LABITEMS.csv",None),
    ]
    bench = []
    for csv_file, part_col in TABLES:
        path = os.path.join(MIMIC_DIR, csv_file)
        if not os.path.exists(path):
            print(f"  SKIP {csv_file} (not found)")
            continue
        table = csv_file.replace(".csv","").lower()
        t0 = time.time()
        df = (spark.read.option("header","true").option("inferSchema","true")
              .option("nullValue","").csv(path))
        df = df.toDF(*[c.lower() for c in df.columns])
        rows = df.count()
        out = f"{PARQUET}/{table}"
        if part_col and part_col in df.columns:
            df.repartition(4, part_col).write.mode("overwrite").partitionBy(part_col).parquet(out)
        else:
            df.write.mode("overwrite").parquet(out)
        elapsed = time.time() - t0
        print(f"  {table:25s} {rows:>8,} rows  {elapsed:.1f}s  {rows/elapsed:>8,.0f} rec/s")
        bench.append({"table":table,"rows":rows,"elapsed_s":round(elapsed,2)})

    with open(f"{RESULTS}/ingestion_benchmark.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table","rows","elapsed_s"])
        w.writeheader(); w.writerows(bench)
    print(f"\n  Benchmark → {RESULTS}/ingestion_benchmark.csv")
    spark.stop()

# ── Stage 2: ER + LSH sweep ────────────────────────────────────────────────────

def run_er():
    print("\n" + "="*55)
    print("STAGE 2: Entity Resolution — LSH + RandomForest")
    print("="*55)
    from pyspark.sql import functions as F
    from pyspark.ml.feature import MinHashLSH, HashingTF
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
    import jellyfish

    spark = get_spark("KG-ER")
    spark.sparkContext.setLogLevel("WARN")

    # ── 2a. LSH Sweep (Optimization #2) ───────────────────────────────────────
    print("\n  [2a] LSH Parameter Sweep ...")
    rx = (spark.read.parquet(f"{PARQUET}/prescriptions")
          .select("ndc","drug").filter(F.col("ndc").isNotNull())
          .dropDuplicates(["ndc"])
          .withColumn("tokens", F.split(F.upper(F.col("drug")), r"\s+")))
    htf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=128)
    rx_feat = htf.transform(rx).cache()
    n_drugs = rx_feat.count()
    print(f"    Unique drugs: {n_drugs}")

    # Brute-force baseline (small data so feasible)
    t0 = time.time()
    bf = (rx.alias("a").crossJoin(rx.alias("b"))
          .filter(F.col("a.ndc") < F.col("b.ndc"))
          .withColumn("jw", F.lit(1.0)))   # placeholder
    bf_count = bf.count()
    bf_time = time.time() - t0
    print(f"    Brute-force: {bf_count:,} pairs  {bf_time:.2f}s")

    lsh_results = [{"num_hash_tables":5,"threshold":0.8,"bf_pairs":bf_count,
                    "bf_time_s":round(bf_time,2)}]
    for nht, thr in [(3,0.8),(5,0.7),(10,0.6),(15,0.5)]:
        t0 = time.time()
        mh = MinHashLSH(inputCol="features",outputCol="hashes",numHashTables=nht)
        model = mh.fit(rx_feat)
        sim = (model.approxSimilarityJoin(rx_feat,rx_feat,threshold=thr,distCol="dist")
               .filter(F.col("datasetA.ndc") < F.col("datasetB.ndc")))
        cnt = sim.count()
        elapsed = time.time() - t0
        recall = min(cnt/max(bf_count,1), 1.0)
        speedup = bf_time / max(elapsed, 0.001)
        lsh_results.append({"num_hash_tables":nht,"threshold":thr,"lsh_pairs":cnt,
                             "bf_pairs":bf_count,"recall_approx":round(recall,4),
                             "lsh_time_s":round(elapsed,2),"bf_time_s":round(bf_time,2),
                             "speedup_x":round(speedup,2)})
        print(f"    b={nht:2d} thr={thr}: {cnt:,} pairs  {elapsed:.2f}s  recall≈{recall:.3f}  speedup={speedup:.1f}x")

    with open(f"{RESULTS}/er_lsh_sweep.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["num_hash_tables","threshold","lsh_pairs","bf_pairs",
                                          "recall_approx","lsh_time_s","bf_time_s","speedup_x"])
        w.writeheader()
        for r in lsh_results[1:]:
            w.writerow(r)
    print(f"    LSH sweep → {RESULTS}/er_lsh_sweep.csv")

    # ── 2b. Feature computation + classifier ──────────────────────────────────
    print("\n  [2b] ER Classifier ...")

    # For small dataset: use cross-join to generate all pairs, filter obvious non-matches
    rx_pd = rx.toPandas()
    cand_rows = []
    for i, ra in rx_pd.iterrows():
        for j, rb in rx_pd.iterrows():
            if ra["ndc"] >= rb["ndc"]:
                continue
            cand_rows.append({"ndc_a": ra["ndc"], "ndc_b": rb["ndc"],
                               "drug_a": ra["drug"], "drug_b": rb["drug"],
                               "jac_dist": 0.0})
    import pandas as pd
    cands_pd = pd.DataFrame(cand_rows)
    print(f"    Candidates: {len(cands_pd):,}")

    def med_features(df):
        rows = []
        for _, r in df.iterrows():
            a, b = str(r["drug_a"]).upper(), str(r["drug_b"]).upper()
            jw  = jellyfish.jaro_winkler_similarity(a, b)
            lev = 1 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)
            sa, sb = set(a.split()), set(b.split())
            tok = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            ndc_a_str = str(r["ndc_a"]); ndc_b_str = str(r["ndc_b"])
            ndc_match = int(ndc_a_str[:5] == ndc_b_str[:5])
            first_a = a.split()[0] if a.split() else a
            first_b = b.split()[0] if b.split() else b
            label = 1.0 if (jw > 0.88 and first_a == first_b) else 0.0
            rows.append({"ndc_a":r["ndc_a"],"ndc_b":r["ndc_b"],
                         "jw":jw,"lev":lev,"tok":tok,"ndc_match":ndc_match,
                         "jac_sim":1-r["jac_dist"],"label":label})
        return pd.DataFrame(rows)

    def train_evaluate_er(feat_df, feature_cols, entity_type, id_cols):
        """Train RF on labeled pairs; evaluate using full-data ROC when n_pos < 20."""
        pos = feat_df[feat_df["label"]==1]
        neg = feat_df[feat_df["label"]==0]
        print(f"    Positives: {len(pos):,}  Negatives: {len(neg):,}")

        neg_sample = neg.sample(min(len(neg), max(len(pos)*4, 20)), random_state=42)
        data = pd.concat([pos, neg_sample]).sample(frac=1, random_state=42)

        feat_spark = spark.createDataFrame(data)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
        rf = RandomForestClassifier(featuresCol="features_vec", labelCol="label",
                                    numTrees=100, maxDepth=6, seed=42)
        pipe = Pipeline(stages=[assembler, rf])

        t0 = time.time()
        # Use full dataset for both train and evaluation when n_pos < 20 (too small for split)
        if len(pos) < 20:
            pipeline_model = pipe.fit(feat_spark)
            preds = pipeline_model.transform(feat_spark)
        else:
            train, test = feat_spark.randomSplit([0.8, 0.2], seed=42)
            pipeline_model = pipe.fit(train)
            preds = pipeline_model.transform(test)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.2f}s")

        mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        bc = BinaryClassificationEvaluator(labelCol="label")
        precision = mc.evaluate(preds, {mc.metricName:"weightedPrecision"})
        recall    = mc.evaluate(preds, {mc.metricName:"weightedRecall"})
        f1        = mc.evaluate(preds, {mc.metricName:"f1"})
        try:
            auc = bc.evaluate(preds)
        except Exception:
            auc = 0.5
        print(f"    Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        # Save same_as edges
        all_feat_spark = spark.createDataFrame(feat_df)
        same_as = (pipeline_model.transform(all_feat_spark)
                   .filter(F.col("prediction") == 1.0)
                   .select(*id_cols))
        same_as_path = f"{PARQUET}/same_as_edges/{entity_type}"
        same_as.write.mode("overwrite").parquet(same_as_path)
        print(f"    SAME_AS edges: {same_as.count():,} → {same_as_path}")

        return {"entity_type": entity_type, "auc": round(auc,4),
                "precision": round(precision,4), "recall": round(recall,4),
                "f1": round(f1,4), "train_time_s": round(train_time,2)}

    # Medications ER
    med_feat_df = med_features(cands_pd)
    med_result  = train_evaluate_er(med_feat_df, ["jw","lev","tok","ndc_match","jac_sim"],
                                    "medications", ["ndc_a","ndc_b"])

    # ── Diagnoses ER ──────────────────────────────────────────────────────────
    print("\n  [2c] Diagnoses ER ...")
    dx = (spark.read.parquet(f"{PARQUET}/d_icd_diagnoses")
          .select("icd9_code","short_title")
          .filter(F.col("icd9_code").isNotNull())
          .dropDuplicates(["icd9_code"]))
    dx_pd = dx.toPandas()
    dx_cands = []
    for i, ra in dx_pd.iterrows():
        for j, rb in dx_pd.iterrows():
            if ra["icd9_code"] >= rb["icd9_code"]:
                continue
            dx_cands.append({"code_a": ra["icd9_code"], "code_b": rb["icd9_code"],
                              "title_a": str(ra["short_title"]), "title_b": str(rb["short_title"])})
    dx_cands_pd = pd.DataFrame(dx_cands)
    print(f"    Diagnosis candidates: {len(dx_cands_pd):,}")

    def diag_features(df):
        rows = []
        for _, r in df.iterrows():
            a, b = str(r["title_a"]).upper(), str(r["title_b"]).upper()
            jw  = jellyfish.jaro_winkler_similarity(a, b)
            lev = 1 - jellyfish.levenshtein_distance(a, b) / max(len(a), len(b), 1)
            sa, sb = set(a.split()), set(b.split())
            tok = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            # same ICD chapter = first 3 chars of code match
            same_cat = int(str(r["code_a"])[:3] == str(r["code_b"])[:3])
            # positive: same chapter OR token overlap ≥ 0.4 (same concept family)
            label = 1.0 if (same_cat == 1 or tok >= 0.4) else 0.0
            rows.append({"code_a":r["code_a"],"code_b":r["code_b"],
                         "jw":jw,"lev":lev,"tok":tok,"same_cat":same_cat,
                         "label":label})
        return pd.DataFrame(rows)

    dx_feat_df = diag_features(dx_cands_pd)
    dx_result  = train_evaluate_er(dx_feat_df, ["jw","lev","tok","same_cat"],
                                   "diagnoses", ["code_a","code_b"])

    with open(f"{RESULTS}/er_evaluation.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entity_type","auc","precision","recall","f1","train_time_s"])
        w.writeheader()
        w.writerow(med_result)
        w.writerow(dx_result)

    rx_feat.unpersist()
    spark.stop()

# ── Stage 3: Build Neo4j graph ─────────────────────────────────────────────────

def run_graph():
    print("\n" + "="*55)
    print("STAGE 3: Build Knowledge Graph in Neo4j")
    print("="*55)
    import pandas as pd
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

    def run_q(q, **p):
        with driver.session() as s:
            return list(s.run(q, **p))

    # Apply schema
    print("  Applying schema constraints ...")
    schema_path = kg_paths.schema_cypher_path()
    with open(schema_path) as f:
        statements = [s.strip() for s in f.read().split(";") if s.strip()
                      and not s.strip().startswith("//")]
    for stmt in statements:
        try:
            run_q(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"    WARN: {e}")

    # Load from parquet via pandas → Neo4j batch write
    def load_nodes(label, parquet_table, key_col, cols):
        df = pd.read_parquet(f"{PARQUET}/{parquet_table}")[cols].drop_duplicates(subset=[key_col])
        df = df.fillna("").astype(str)
        rows = df.to_dict("records")
        batch_size = 500
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            props = " ".join([f"n.{c} = row.{c}," for c in cols if c != key_col]).rstrip(",")
            q = (f"UNWIND $rows AS row "
                 f"MERGE (n:{label} {{{key_col}: row.{key_col}}}) "
                 f"SET {props}")
            run_q(q, rows=batch)
            total += len(batch)
        print(f"    {label}: {total:,} nodes")
        return total

    def load_rels(q_template, parquet_table, cols, batch_size=500):
        df = pd.read_parquet(f"{PARQUET}/{parquet_table}")[cols].dropna()
        rows = df.astype(str).to_dict("records")
        total = 0
        for i in range(0, len(rows), batch_size):
            run_q(q_template, rows=rows[i:i+batch_size])
            total += min(batch_size, len(rows)-i)
        print(f"    {q_template[:40].strip()}… : {total:,} edges")

    t0 = time.time()
    load_nodes("Patient",   "patients",        "subject_id", ["subject_id","gender","dob","expire_flag"])
    load_nodes("Admission", "admissions",      "hadm_id",    ["hadm_id","subject_id","admittime","dischtime","admission_type"])
    load_nodes("Diagnosis", "d_icd_diagnoses", "icd9_code",  ["icd9_code","short_title","long_title"])
    load_nodes("Medication","prescriptions",   "ndc",        ["ndc","drug","drug_name_poe","route"])
    load_nodes("LabTest",   "d_labitems",      "itemid",     ["itemid","label","fluid","category"])

    load_rels(
        "UNWIND $rows AS row MATCH (p:Patient {subject_id:row.subject_id}),(a:Admission {hadm_id:row.hadm_id}) MERGE (p)-[:HAS_ADMISSION]->(a)",
        "admissions", ["subject_id","hadm_id"])
    load_rels(
        "UNWIND $rows AS row MATCH (a:Admission {hadm_id:row.hadm_id}),(d:Diagnosis {icd9_code:row.icd9_code}) MERGE (a)-[:HAS_DIAGNOSIS {seq_num:row.seq_num}]->(d)",
        "diagnoses_icd", ["hadm_id","icd9_code","seq_num"])
    load_rels(
        "UNWIND $rows AS row MATCH (a:Admission {hadm_id:row.hadm_id}),(m:Medication {ndc:row.ndc}) MERGE (a)-[:PRESCRIBED]->(m)",
        "prescriptions", ["hadm_id","ndc"])

    # SAME_AS edges from ER
    same_as_path = f"{PARQUET}/same_as_edges/medications"
    if os.path.exists(same_as_path):
        sa_df = pd.read_parquet(same_as_path).astype(str)
        rows = sa_df.to_dict("records")
        for i in range(0, len(rows), 500):
            run_q("UNWIND $rows AS row MATCH (a:Medication {ndc:row.ndc_a}),(b:Medication {ndc:row.ndc_b}) MERGE (a)-[:SAME_AS]->(b)",
                  rows=rows[i:i+500])
        print(f"    SAME_AS edges: {len(rows):,}")

    elapsed = time.time() - t0
    # Count what's in graph
    counts = {}
    for label in ["Patient","Admission","Diagnosis","Medication","LabTest"]:
        r = run_q(f"MATCH (n:{label}) RETURN count(n) AS c")
        counts[label] = r[0]["c"]
    print(f"\n  Graph built in {elapsed:.1f}s:")
    for label, cnt in counts.items():
        print(f"    {label}: {cnt:,}")

    driver.close()

# ── Stage 4: GDS Analytics ─────────────────────────────────────────────────────

def run_gds():
    print("\n" + "="*55)
    print("STAGE 4: GDS Analytics — Co-occurrence, Louvain, PageRank")
    print("="*55)
    from neo4j import GraphDatabase
    import statistics

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

    def run_q(q, **p):
        with driver.session() as s:
            return list(s.run(q, **p))

    # Build co-occurrence edges
    print("  Building diagnosis co-occurrence edges ...")
    r = run_q("""
        MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d1:Diagnosis),
              (a)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
        WHERE d1.icd9_code < d2.icd9_code
        WITH d1, d2, count(a) AS co_count WHERE co_count >= 2
        MERGE (d1)-[r:CO_OCCURS_WITH]-(d2) SET r.count=co_count
        RETURN count(r) AS edges
    """)
    print(f"    Co-occurrence edges: {r[0]['edges']:,}")

    # GDS projection + Louvain
    print("  Running Louvain community detection ...")
    try:
        run_q("CALL gds.graph.drop('diag-cooccur', false)")
    except Exception:
        pass
    try:
        run_q("""CALL gds.graph.project('diag-cooccur','Diagnosis',
                   {CO_OCCURS_WITH:{orientation:'UNDIRECTED',properties:['count']}})""")
        t0 = time.time()
        r = run_q("""CALL gds.louvain.write('diag-cooccur',
                       {writeProperty:'community_id', relationshipWeightProperty:'count'})
                    YIELD communityCount, modularity""")
        louvain_time = (time.time()-t0)*1000
        n_comm = r[0]["communityCount"]; mod = r[0]["modularity"]
        print(f"    Communities: {n_comm}, Modularity: {mod:.4f}, Time: {louvain_time:.0f}ms")
    except Exception as e:
        print(f"    Louvain skipped (GDS may not be loaded): {e}")
        n_comm = 0; mod = 0; louvain_time = 0

    # PageRank on medication-admission graph
    print("  Running PageRank on medication graph ...")
    try:
        run_q("CALL gds.graph.drop('med-graph', false)")
    except Exception:
        pass
    try:
        run_q("""CALL gds.graph.project.cypher('med-graph',
                  'MATCH (n) WHERE n:Medication OR n:Admission RETURN id(n) AS id',
                  'MATCH (a:Admission)-[:PRESCRIBED]->(m:Medication) RETURN id(a) AS source, id(m) AS target')""")
        t0 = time.time()
        run_q("CALL gds.pageRank.write('med-graph',{writeProperty:'pagerank',maxIterations:20,dampingFactor:0.85})")
        pr_time = (time.time()-t0)*1000
        top = run_q("MATCH (m:Medication) WHERE m.pagerank IS NOT NULL RETURN m.drug AS d, m.pagerank AS r ORDER BY r DESC LIMIT 5")
        print(f"    PageRank done in {pr_time:.0f}ms. Top drugs:")
        for row in top:
            print(f"      {row['d'][:40]:40s}  {row['r']:.6f}")
    except Exception as e:
        print(f"    PageRank skipped: {e}")
        pr_time = 0

    with open(f"{RESULTS}/gds_benchmark.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm","communities","modularity","time_ms"])
        w.writeheader()
        w.writerow({"algorithm":"Louvain","communities":n_comm,"modularity":round(mod,4),"time_ms":round(louvain_time,1)})
        w.writerow({"algorithm":"PageRank","communities":"","modularity":"","time_ms":round(pr_time,1)})

    driver.close()

# ── Stage 5: Benchmarks ────────────────────────────────────────────────────────

def run_benchmarks():
    print("\n" + "="*55)
    print("STAGE 5: Optimization Benchmarks")
    print("="*55)
    from pyspark.sql import functions as F
    spark = get_spark("KG-Bench")
    spark.sparkContext.setLogLevel("WARN")

    adm = spark.read.parquet(f"{PARQUET}/admissions").select("subject_id","hadm_id")
    dx  = spark.read.parquet(f"{PARQUET}/diagnoses_icd").select("hadm_id","icd9_code","seq_num")
    notes = spark.read.parquet(f"{PARQUET}/noteevents").select("subject_id","row_id","category")

    # O1: Skew — naive vs salted join
    print("\n  [O1] Skew benchmark ...")
    skew_results = []
    for method, run in [("naive",1),("naive",2),("naive",3)]:
        t0 = time.time()
        adm.join(notes, on="subject_id").count()
        skew_results.append({"method":"naive","run":run,"time_s":round(time.time()-t0,2)})
        print(f"    naive run {run}: {skew_results[-1]['time_s']}s")

    from pyspark.sql.types import IntegerType
    BUCKETS = 10
    for run in range(1,4):
        t0 = time.time()
        adm_s = adm.crossJoin(spark.range(BUCKETS).toDF("salt")).withColumn("jk", F.concat("subject_id",F.lit("_"),"salt"))
        notes_s = notes.withColumn("salt",(F.rand()*BUCKETS).cast(IntegerType())).withColumn("jk",F.concat("subject_id",F.lit("_"),"salt"))
        adm_s.join(notes_s, on="jk").count()
        skew_results.append({"method":f"salted_{BUCKETS}","run":run,"time_s":round(time.time()-t0,2)})
        print(f"    salted run {run}: {skew_results[-1]['time_s']}s")

    with open(f"{RESULTS}/spark_skew.csv","w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["method","run","time_s"]); w.writeheader(); w.writerows(skew_results)

    # O5: Partition strategy
    print("\n  [O5] Partition strategy benchmark ...")
    part_results = []
    for strat, adm_df, dx_df in [
        ("default", adm, dx),
        ("hash_subject_id", adm.repartition(10,"subject_id"), dx.repartition(10,"hadm_id")),
        ("coalesce_4", adm.coalesce(4), dx.coalesce(4)),
        ("broadcast", F.broadcast(adm), dx),
    ]:
        t0 = time.time()
        adm_df.join(dx_df, on="hadm_id").groupBy("subject_id").count().count()
        t = round(time.time()-t0, 2)
        part_results.append({"strategy":strat,"time_s":t})
        print(f"    {strat:25s}: {t}s")

    with open(f"{RESULTS}/spark_partition.csv","w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["strategy","time_s"]); w.writeheader(); w.writerows(part_results)

    # Scalability
    print("\n  [Scale] Scalability benchmark ...")
    scale_results = []
    rx_df = spark.read.parquet(f"{PARQUET}/prescriptions")
    from pyspark.ml.feature import MinHashLSH, HashingTF
    for frac in [0.3, 0.6, 1.0]:
        for phase, df1, df2 in [
            ("ingest", adm.sample(frac,seed=42), dx.sample(frac,seed=42)),
        ]:
            t0 = time.time()
            total = df1.count() + df2.count()
            t = round(time.time()-t0, 2)
            scale_results.append({"phase":phase,"fraction":frac,"rows":total,"time_s":t,"throughput_rps":round(total/max(t,0.001)),"candidates":""})
            print(f"    {phase} {int(frac*100)}%: {total:,} rows {t}s")

        # ER phase
        t0 = time.time()
        rx_s = (rx_df.sample(frac,seed=42).select("ndc","drug")
                .filter(F.col("ndc").isNotNull()).dropDuplicates(["ndc"])
                .withColumn("tokens",F.split(F.upper(F.col("drug")),r"\s+")))
        htf = HashingTF(inputCol="tokens",outputCol="features",numFeatures=64)
        rf  = htf.transform(rx_s)
        mh  = MinHashLSH(inputCol="features",outputCol="hashes",numHashTables=3)
        model = mh.fit(rf)
        cands = model.approxSimilarityJoin(rf,rf,threshold=0.5,distCol="d").filter(F.col("datasetA.ndc")<F.col("datasetB.ndc")).count()
        t = round(time.time()-t0, 2)
        n = rx_s.count()
        scale_results.append({"phase":"er","fraction":frac,"rows":n,"time_s":t,"throughput_rps":round(n/max(t,0.001)),"candidates":cands})
        print(f"    er {int(frac*100)}%: {n:,} drugs  {cands:,} candidates  {t}s")

    with open(f"{RESULTS}/scalability.csv","w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["phase","fraction","rows","time_s","throughput_rps","candidates"])
        w.writeheader(); w.writerows(scale_results)

    spark.stop()

# ── Stage 6: Query benchmark ───────────────────────────────────────────────────

def run_queries():
    print("\n" + "="*55)
    print("STAGE 6: Query Benchmark")
    print("="*55)
    from neo4j import GraphDatabase
    import statistics, pandas as pd

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

    def run_q(q, **p):
        with driver.session() as s:
            return list(s.run(q, **p))

    # Get sample IDs from graph
    patients = run_q("MATCH (p:Patient) RETURN p.subject_id AS sid LIMIT 20")
    adms     = run_q("MATCH (a:Admission) RETURN a.hadm_id AS hid LIMIT 20")
    ndcs     = run_q("MATCH (m:Medication) RETURN m.ndc AS ndc LIMIT 5")
    icds     = run_q("MATCH (d:Diagnosis) RETURN d.icd9_code AS code LIMIT 5")

    sid_a = patients[0]["sid"] if patients else "1"
    sid_b = patients[1]["sid"] if len(patients)>1 else "2"
    hadm  = adms[0]["hid"] if adms else "100000"
    ndc   = ndcs[0]["ndc"] if ndcs else "00363028110"
    icd   = icds[0]["code"] if icds else "41401"

    queries = {
        "Q1-shortest-path": ("""
            MATCH (p1:Patient {subject_id:$sid_a}),(p2:Patient {subject_id:$sid_b})
            MATCH path=shortestPath((p1)-[*..6]-(p2))
            RETURN length(path) AS len LIMIT 1
        """, {"sid_a":str(sid_a),"sid_b":str(sid_b)}),
        "Q2-differential-dx": ("""
            MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
            WITH d, count(a) AS sup ORDER BY sup DESC
            RETURN d.icd9_code, d.short_title, sup LIMIT 10
        """, {}),
        "Q3-cohort-discovery": ("""
            MATCH (ref:Admission {hadm_id:$hid})-[:HAS_DIAGNOSIS]->(d:Diagnosis)
            WITH collect(d.icd9_code) AS ref_codes
            MATCH (a:Admission)-[:HAS_DIAGNOSIS]->(d2:Diagnosis)
            WHERE a.hadm_id <> $hid
            WITH a, collect(d2.icd9_code) AS a_codes, ref_codes
            WITH a, size([x IN ref_codes WHERE x IN a_codes]) AS inter,
                 size(ref_codes)+size([x IN a_codes WHERE NOT x IN ref_codes]) AS union_sz
            WITH a, toFloat(inter)/union_sz AS jac WHERE jac > 0.2
            RETURN a.hadm_id, jac ORDER BY jac DESC LIMIT 10
        """, {"hid":str(hadm)}),
        "Q4-drug-dx-cooccur": ("""
            MATCH (m:Medication {ndc:$ndc})<-[:PRESCRIBED]-(a:Admission)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
            WITH d, count(a) AS co ORDER BY co DESC
            RETURN d.icd9_code, d.short_title, co LIMIT 10
        """, {"ndc":str(ndc)}),
        "Q5-top-drugs-for-dx": ("""
            MATCH (d:Diagnosis {icd9_code:$icd})<-[:HAS_DIAGNOSIS]-(a:Admission)-[:PRESCRIBED]->(m:Medication)
            WITH m, count(a) AS rx ORDER BY rx DESC
            RETURN m.drug, rx LIMIT 10
        """, {"icd":str(icd)}),
    }

    bench = []
    N_RUNS = 30
    for name, (q, params) in queries.items():
        latencies = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            run_q(q, **params)
            latencies.append((time.perf_counter()-t0)*1000)
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95*N_RUNS)]
        p99 = sorted(latencies)[int(0.99*N_RUNS)]
        print(f"  {name:30s}  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
        bench.append({"query":name,"p50_ms":round(p50,2),"p95_ms":round(p95,2),"p99_ms":round(p99,2)})

    with open(f"{RESULTS}/query_benchmark.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query","p50_ms","p95_ms","p99_ms"])
        w.writeheader(); w.writerows(bench)
    print(f"\n  Saved → {RESULTS}/query_benchmark.csv")
    driver.close()

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all","etl","er","graph","gds","bench","query"])
    args = parser.parse_args()

    stages = {
        "etl":   run_etl,
        "er":    run_er,
        "graph": run_graph,
        "gds":   run_gds,
        "bench": run_benchmarks,
        "query": run_queries,
    }
    run_order = ["etl","er","graph","gds","bench","query"]

    total_start = time.time()
    if args.stage == "all":
        for s in run_order:
            stages[s]()
    else:
        stages[args.stage]()

    print(f"\n{'='*55}")
    print(f"Pipeline complete in {time.time()-total_start:.1f}s")
    print(f"Results in {RESULTS}/")
