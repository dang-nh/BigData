"""
Generate synthetic MIMIC-III-compatible data for pipeline development/demo.
Produces realistic clinical data: ~500 patients, ~2000 admissions, ~50k notes.
"""

import os
import csv
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

OUT_DIR = Path(os.getenv("MIMIC_DATA_DIR", "/home/ntan/final-proj/data/mimic-iii"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PATIENTS    = 500
N_ADMISSIONS  = 2000
N_NOTES       = 8000
N_LABEVENTS   = 30000

# ── Reference data ─────────────────────────────────────────────────────────────

ICD9_DIAGNOSES = [
    ("41401","Crnry athrscl natve vssl"),("4280","CHF NOS"),("42731","Atrial fibrillation"),
    ("51881","Acute respiratry failure"),("99591","Sepsis"),("25000","Dmii wo cmp nt st uncntr"),
    ("4019","Hypertension NOS"),("5849","Acute renal failure NOS"),("486","Pneumonia organism NOS"),
    ("2724","Hyperlipidemia NOS"),("43491","Cereb art ocl NOS w infrc"),("53081","Esophageal reflux"),
    ("7802","Syncope and collapse"),("78552","Septic shock"),("40390","Hyp kid NOS w/o cr kid"),
    ("2761","Hyposmolality"),("2762","Acidosis"),("27651","Dehydration"),
    ("5070","Food/vomit pneumonitis"),("0389","Unspecified septicemia"),
    ("99811","Hemorrhage compl proc"),("V5861","Long-term use anticoags"),
    ("3051","Tobacco use disorder"),("30500","Alcohol abuse-unspec"),
    ("7840","Headache"),("78659","Other chest pain"),("7891","Abdominal pain NOS"),
]

ICD9_PROCEDURES = [
    ("9671","Cont inv mec ven 96+ hr"),("3893","Venous cath NEC"),
    ("3731","Pericardiocentesis"),("9604","Insert endotracheal tube"),
    ("8856","Cor arteriography NEC"),("3615","Single int mam-LAD bypass"),
    ("0066","Percut translum cor ang"),("9915","Parenteral infusion NEC"),
    ("4553","Oth partial colectomy"),("5123","Laparoscopic cholecystect"),
]

DRUGS = [
    ("Aspirin","00363028110","C0004057"),("Heparin Sodium","00641040025","C0019134"),
    ("Metoprolol Tartrate","00781517210","C0025859"),("Furosemide","00703427003","C0016860"),
    ("Lisinopril","68382003601","C0065374"),("Atorvastatin","00069015530","C0286651"),
    ("Potassium Chloride","00338005302","C0032821"),("Pantoprazole","00008408102","C0081876"),
    ("Warfarin Sodium","00056017975","C0043031"),("Insulin Regular","00002821501","C0021641"),
    ("Vancomycin HCl","00143317910","C0042313"),("Piperacillin-Tazobactam","00206383546","C0718413"),
    ("Norepinephrine","00409478102","C0028351"),("Morphine Sulfate","00641606125","C0026549"),
    ("Lorazepam","00641606175","C0024069"),("Dextrose 5%","00338001702","C0012554"),
    ("Normal Saline 0.9%","00338004904","C0036082"),("Amiodarone HCl","66220015601","C0002598"),
    ("Metronidazole","00338153648","C0025872"),("Ceftriaxone","00143992401","C0007561"),
    ("Aspirin 81mg","00363027971","C0004057"),("Heparin Flush","00641040010","C0019134"),
    ("Metoprolol Succinate","00378712105","C0025859"),("Furosemide IV","00703427010","C0016860"),
]

LAB_ITEMS = [
    (50818,"pCO2","Blood","Blood Gas"),
    (50820,"pH","Blood","Blood Gas"),
    (50821,"pO2","Blood","Blood Gas"),
    (50912,"Creatinine","Blood","Chemistry"),
    (50971,"Potassium","Blood","Chemistry"),
    (50983,"Sodium","Blood","Chemistry"),
    (51006,"Urea Nitrogen","Blood","Chemistry"),
    (51222,"Hemoglobin","Blood","Hematology"),
    (51265,"Platelet Count","Blood","Hematology"),
    (51279,"Red Blood Cells","Blood","Hematology"),
    (51301,"White Blood Cells","Blood","Hematology"),
    (50902,"Chloride","Blood","Chemistry"),
    (50931,"Glucose","Blood","Chemistry"),
    (50960,"Magnesium","Blood","Chemistry"),
    (50893,"Calcium Total","Blood","Chemistry"),
]

NOTE_TEMPLATES = [
    "Patient is a {age}-year-old {gender} with a history of {hx} presenting with {cc}. "
    "Vital signs: BP {bp}, HR {hr}, RR {rr}, T {temp}. "
    "Physical exam notable for {exam}. Labs significant for {labs}. "
    "Assessment: {dx}. Plan: {plan}.",
    "CHIEF COMPLAINT: {cc}.\nHISTORY: {age} y/o {gender} with {hx}.\n"
    "PHYSICAL EXAM: {exam}.\nLABS: {labs}.\nIMPRESSION: {dx}.\nPLAN: {plan}.",
    "Seen in ED for {cc}. PMH: {hx}. "
    "Exam: {exam}. "
    "Diagnosis: {dx}. Initiated {plan}.",
]
CC = ["chest pain","shortness of breath","altered mental status","fever and chills",
      "acute onset dyspnea","lower extremity edema","syncope","abdominal pain",
      "weakness and fatigue","palpitations"]
HX = ["hypertension and diabetes","CAD and CHF","COPD and HTN","CKD and anemia",
      "atrial fibrillation on anticoagulation","prior CABG","DM type 2 and hyperlipidemia"]
EXAM = ["bilateral crackles","2+ pitting edema","JVD present","irregular heart rate",
        "diffuse wheezing","soft abdomen with mild tenderness","decreased breath sounds"]
LABS = ["elevated troponin","BNP markedly elevated","WBC 14.5 with left shift",
        "Cr 2.1 from baseline 1.0","lactate 3.2","Na 128 hyponatremia","Hgb 7.2 anemia"]
DX_TEXT = ["STEMI","Acute decompensated heart failure","Community acquired pneumonia",
           "Sepsis secondary to UTI","COPD exacerbation","Atrial fibrillation with RVR",
           "Acute kidney injury","Hypertensive urgency"]
PLAN_TEXT = ["heparin and cardiology consult","IV furosemide diuresis","broad-spectrum antibiotics",
             "fluid resuscitation and cultures","nebulizers and steroids","rate control and anticoagulation",
             "nephrology consult and IVF","nicardipine drip"]


def rand_date(start=datetime(2100,1,1), end=datetime(2201,1,1)):
    return start + timedelta(seconds=random.randint(0, int((end-start).total_seconds())))


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {path.name}")


def generate():
    print("Generating synthetic MIMIC-III data ...")

    # PATIENTS
    patients = []
    for i in range(1, N_PATIENTS+1):
        dob = rand_date(datetime(2030,1,1), datetime(2080,1,1))
        gender = random.choice(["M","F"])
        expire = random.random() < 0.15
        dod = fmt(rand_date(datetime(2180,1,1))) if expire else ""
        patients.append({"ROW_ID":i,"SUBJECT_ID":i,"GENDER":gender,
                         "DOB":fmt(dob),"DOD":dod,"DOD_HOSP":"","DOD_SSN":"",
                         "EXPIRE_FLAG":int(expire)})
    write_csv(OUT_DIR/"PATIENTS.csv", list(patients[0].keys()), patients)

    # ADMISSIONS
    adm_id = 100000
    admissions = []
    adm_types = ["EMERGENCY","ELECTIVE","URGENT","NEWBORN"]
    for pid in range(1, N_PATIENTS+1):
        n_adm = random.randint(1, 6)
        admit_time = rand_date(datetime(2150,1,1), datetime(2190,1,1))
        for _ in range(n_adm):
            discharge = admit_time + timedelta(days=random.randint(1,20),
                                               hours=random.randint(0,23))
            icd = random.choice(ICD9_DIAGNOSES)[1]
            admissions.append({
                "ROW_ID": adm_id, "SUBJECT_ID": pid, "HADM_ID": adm_id,
                "ADMITTIME": fmt(admit_time), "DISCHTIME": fmt(discharge),
                "DEATHTIME": "", "ADMISSION_TYPE": random.choice(adm_types),
                "ADMISSION_LOCATION": "EMERGENCY ROOM ADMIT",
                "DISCHARGE_LOCATION": "HOME",
                "INSURANCE": random.choice(["Medicare","Medicaid","Private","Government"]),
                "LANGUAGE": "ENGL", "RELIGION": "", "MARITAL_STATUS": "",
                "ETHNICITY": "WHITE", "EDREGTIME": "", "EDOUTTIME": "",
                "DIAGNOSIS": icd, "HOSPITAL_EXPIRE_FLAG": 0,
                "HAS_CHARTEVENTS_DATA": 1,
            })
            admit_time = discharge + timedelta(days=random.randint(30, 500))
            adm_id += 1
            if len(admissions) >= N_ADMISSIONS:
                break
        if len(admissions) >= N_ADMISSIONS:
            break
    write_csv(OUT_DIR/"ADMISSIONS.csv", list(admissions[0].keys()), admissions)

    # D_ICD_DIAGNOSES
    d_diag = [{"ROW_ID":i+1,"ICD9_CODE":c,"SHORT_TITLE":t,"LONG_TITLE":t+" (long)"}
              for i,(c,t) in enumerate(ICD9_DIAGNOSES)]
    write_csv(OUT_DIR/"D_ICD_DIAGNOSES.csv", list(d_diag[0].keys()), d_diag)

    # D_ICD_PROCEDURES
    d_proc = [{"ROW_ID":i+1,"ICD9_CODE":c,"SHORT_TITLE":t,"LONG_TITLE":t}
              for i,(c,t) in enumerate(ICD9_PROCEDURES)]
    write_csv(OUT_DIR/"D_ICD_PROCEDURES.csv", list(d_proc[0].keys()), d_proc)

    # DIAGNOSES_ICD (per-admission)
    diag_rows = []
    for adm in admissions:
        n_dx = random.randint(1, 8)
        codes = random.sample(ICD9_DIAGNOSES, min(n_dx, len(ICD9_DIAGNOSES)))
        for seq, (code, _) in enumerate(codes, 1):
            diag_rows.append({"ROW_ID": len(diag_rows)+1,
                              "SUBJECT_ID": adm["SUBJECT_ID"],
                              "HADM_ID": adm["HADM_ID"],
                              "SEQ_NUM": seq, "ICD9_CODE": code})
    write_csv(OUT_DIR/"DIAGNOSES_ICD.csv", list(diag_rows[0].keys()), diag_rows)

    # PROCEDURES_ICD
    proc_rows = []
    for adm in admissions:
        if random.random() < 0.4:
            n_proc = random.randint(1, 3)
            codes = random.sample(ICD9_PROCEDURES, min(n_proc, len(ICD9_PROCEDURES)))
            for seq, (code, _) in enumerate(codes, 1):
                proc_rows.append({"ROW_ID": len(proc_rows)+1,
                                  "SUBJECT_ID": adm["SUBJECT_ID"],
                                  "HADM_ID": adm["HADM_ID"],
                                  "SEQ_NUM": seq, "ICD9_CODE": code})
    write_csv(OUT_DIR/"PROCEDURES_ICD.csv", list(proc_rows[0].keys()), proc_rows)

    # PRESCRIPTIONS
    rx_rows = []
    for adm in admissions:
        n_rx = random.randint(2, 10)
        drugs = random.sample(DRUGS, min(n_rx, len(DRUGS)))
        for drug, ndc, _ in drugs:
            # Introduce some dirty variants for ER
            drug_name = drug
            if random.random() < 0.15:
                drug_name = drug.upper()
            elif random.random() < 0.10:
                drug_name = drug.replace(" ", "-")
            start = datetime.strptime(adm["ADMITTIME"], "%Y-%m-%d %H:%M:%S")
            rx_rows.append({
                "ROW_ID": len(rx_rows)+1,
                "SUBJECT_ID": adm["SUBJECT_ID"],
                "HADM_ID": adm["HADM_ID"],
                "ICUSTAY_ID": "",
                "STARTDATE": fmt(start),
                "ENDDATE": fmt(start + timedelta(days=random.randint(1,10))),
                "DRUG_TYPE": "MAIN",
                "DRUG": drug_name,
                "DRUG_NAME_POE": drug,
                "DRUG_NAME_GENERIC": drug,
                "FORMULARY_DRUG_CD": "".join(random.choices(string.ascii_uppercase, k=6)),
                "GSN": "", "NDC": ndc,
                "PROD_STRENGTH": f"{random.choice([5,10,25,50,100,250,500])} mg",
                "DOSE_VAL_RX": str(random.choice([1,2,5,10,25,50,100,250])),
                "DOSE_UNIT_RX": random.choice(["mg","mEq","units","mcg"]),
                "FORM_VAL_DISP": "1",
                "FORM_UNIT_DISP": random.choice(["TAB","CAP","VIAL","AMP"]),
                "ROUTE": random.choice(["PO","IV","SC","IM","SL"]),
            })
    write_csv(OUT_DIR/"PRESCRIPTIONS.csv", list(rx_rows[0].keys()), rx_rows)

    # D_LABITEMS
    d_lab = [{"ROW_ID":i+1,"ITEMID":it,"LABEL":lb,"FLUID":fl,"CATEGORY":cat,"LOINC_CODE":""}
             for i,(it,lb,fl,cat) in enumerate(LAB_ITEMS)]
    write_csv(OUT_DIR/"D_LABITEMS.csv", list(d_lab[0].keys()), d_lab)

    # LABEVENTS
    lab_rows = []
    lab_values = {50818:(35,45),50820:(7.35,7.45),50821:(80,100),50912:(0.6,1.2),
                  50971:(3.5,5.0),50983:(136,145),51006:(7,20),51222:(12,17),
                  51265:(150,400),51279:(4.2,5.9),51301:(4.5,11.0),
                  50902:(98,107),50931:(70,110),50960:(1.7,2.3),50893:(8.5,10.5)}
    for _ in range(N_LABEVENTS):
        adm = random.choice(admissions)
        item = random.choice(LAB_ITEMS)
        lo, hi = lab_values.get(item[0], (0, 100))
        val = round(random.uniform(lo * 0.7, hi * 1.3), 2)
        abnormal = "H" if val > hi else ("L" if val < lo else "")
        lab_rows.append({
            "ROW_ID": len(lab_rows)+1,
            "SUBJECT_ID": adm["SUBJECT_ID"],
            "HADM_ID": adm["HADM_ID"],
            "ITEMID": item[0],
            "CHARTTIME": adm["ADMITTIME"],
            "VALUE": str(val),
            "VALUENUM": val,
            "VALUEUOM": "mEq/L",
            "FLAG": abnormal,
        })
    write_csv(OUT_DIR/"LABEVENTS.csv", list(lab_rows[0].keys()), lab_rows)

    # NOTEEVENTS
    note_rows = []
    note_categories = ["Discharge summary","Nursing","Physician","Radiology","ECG","Echo"]
    for i in range(N_NOTES):
        adm = random.choice(admissions)
        age = random.randint(45, 90)
        gender = "male" if random.random() > 0.5 else "female"
        template = random.choice(NOTE_TEMPLATES)
        text = template.format(
            age=age, gender=gender,
            hx=random.choice(HX), cc=random.choice(CC),
            bp=f"{random.randint(90,180)}/{random.randint(50,110)}",
            hr=random.randint(55,130), rr=random.randint(12,30),
            temp=round(random.uniform(36.0,39.5),1),
            exam=random.choice(EXAM), labs=random.choice(LABS),
            dx=random.choice(DX_TEXT), plan=random.choice(PLAN_TEXT),
        )
        note_rows.append({
            "ROW_ID": i+1,
            "SUBJECT_ID": adm["SUBJECT_ID"],
            "HADM_ID": adm["HADM_ID"],
            "CHARTDATE": adm["ADMITTIME"][:10],
            "CHARTTIME": adm["ADMITTIME"],
            "STORETIME": adm["ADMITTIME"],
            "CATEGORY": random.choice(note_categories),
            "DESCRIPTION": "Report",
            "CGID": random.randint(1,100),
            "ISERROR": "",
            "TEXT": text,
        })
    write_csv(OUT_DIR/"NOTEEVENTS.csv", list(note_rows[0].keys()), note_rows)

    # CAREGIVERS
    cg_rows = [{"ROW_ID":i,"CGID":i,"LABEL":random.choice(["RN","MD","NP","PA","RT"])}
               for i in range(1,101)]
    write_csv(OUT_DIR/"CAREGIVERS.csv", list(cg_rows[0].keys()), cg_rows)

    print(f"\nDone. Data written to {OUT_DIR}")
    print(f"  Patients: {N_PATIENTS}, Admissions: {len(admissions)}, "
          f"Notes: {N_NOTES}, LabEvents: {N_LABEVENTS}")


if __name__ == "__main__":
    generate()
