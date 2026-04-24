# features.py — feature engineering
from diabetes import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── features.py — add this BEFORE the drop_cols section ─────────────────────

def map_diag_to_category(code):
    """Map ICD-9 diagnosis codes to broad disease categories."""
    try:
        code = str(code).strip()
        # Handle V and E codes
        if code.startswith('V'):
            return 'Other'
        if code.startswith('E'):
            return 'Injury'
        num = float(code)
        if 390 <= num <= 459 or num == 785:
            return 'Circulatory'
        elif 460 <= num <= 519 or num == 786:
            return 'Respiratory'
        elif 520 <= num <= 579 or num == 787:
            return 'Digestive'
        elif 250 <= num <= 250.99:
            return 'Diabetes'
        elif 800 <= num <= 999:
            return 'Injury'
        elif 710 <= num <= 739:
            return 'Musculoskeletal'
        elif 580 <= num <= 629 or num == 788:
            return 'Genitourinary'
        elif 140 <= num <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'
    
def build_features():
    df = load_data()

    # ── 1. Drop columns that won't help the model ────────────────────────────
    df['diag_1_cat'] = df['diag_1'].apply(map_diag_to_category)
    df['diag_2_cat'] = df['diag_2'].apply(map_diag_to_category)
    df['diag_3_cat'] = df['diag_3'].apply(map_diag_to_category)

    # Flag if primary diagnosis is diabetes-related
    df['primary_diag_diabetes'] = (df['diag_1_cat'] == 'Diabetes').astype(int)

    # Flag circulatory as a high-risk comorbidity
    df['has_circulatory'] = (
        (df['diag_1_cat'] == 'Circulatory') |
        (df['diag_2_cat'] == 'Circulatory') |
        (df['diag_3_cat'] == 'Circulatory')
    ).astype(int)

    drop_cols = [
        'weight',           # 97% missing
        'payer_code',       # 40% missing, not clinically meaningful
        'gender',           # not significant in chi-square
        'encounter_id',     # just an ID
        'patient_nbr',      # just an ID
        'readmitted',       # replaced by binary target
        'diag_1',           # raw ICD codes — too granular without grouping
        'diag_2',
        'diag_3',
        'max_glu_serum',    # 95% missing
    ]
    df = df.drop(columns=drop_cols)

    # ── 2. Handle missing values ─────────────────────────────────────────────
    df['race']              = df['race'].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    df['A1Cresult']         = df['A1Cresult'].fillna('Not_tested')

    # ── 3. Convert A1Cresult to a simple flag ────────────────────────────────
    # Was A1C tested or not? (83% missing — treat as binary)
    df['A1C_tested'] = (df['A1Cresult'] != 'Not_tested').astype(int)
    df = df.drop(columns=['A1Cresult'])

    # ── 3b. Map discharge disposition to meaningful groups ───────────────────────
    discharge_map = {
        1:  'Home',
        2:  'Transfer_hospital',
        3:  'Transfer_SNF',           # Skilled Nursing Facility
        4:  'Transfer_ICF',           # Intermediate Care Facility
        5:  'Transfer_inpatient',
        6:  'Home_health_service',
        7:  'Left_AMA',               # Against Medical Advice — high risk
        8:  'Home_IV_care',
        9:  'Inpatient_admitted',
        10: 'Transfer_neonatal',
        11: 'Expired',
        12: 'Outpatient_return',
        13: 'Hospice_home',
        14: 'Hospice_facility',
        15: 'Swing_bed',
        16: 'Transfer_outpatient',
        17: 'Transfer_outpatient',
        18: 'Unknown',
        19: 'Expired',
        20: 'Expired',
        21: 'Expired',
        22: 'Transfer_rehab',
        23: 'Transfer_longterm',
        24: 'Transfer_SNF',
        25: 'Unknown',
        26: 'Unknown',
        27: 'Transfer_federal',
        28: 'Transfer_psych',
        29: 'Transfer_hospital',
        30: 'Transfer_other'
    }

    # ── 3c. Map admission type to meaningful groups ──────────────────────────────
    admission_type_map = {
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective',
        4: 'Newborn',
        5: 'Unknown',
        6: 'Unknown',
        7: 'Trauma',
        8: 'Unknown'
    }

    # ── 3d. Map admission source to meaningful groups ────────────────────────────
    admission_source_map = {
        1:  'Physician_referral',
        2:  'Clinic_referral',
        3:  'HMO_referral',
        4:  'Transfer_hospital',
        5:  'Transfer_SNF',
        6:  'Transfer_other',
        7:  'Emergency_room',
        8:  'Court_law',
        9:  'Unknown',
        10: 'Transfer_other',
        11: 'Normal_delivery',
        12: 'Premature_delivery',
        13: 'Sick_baby',
        14: 'Extramural_birth',
        15: 'Unknown',
        17: 'Unknown',
        18: 'Transfer_other',
        19: 'Unknown',
        20: 'Unknown',
        21: 'Unknown',
        22: 'Unknown',
        23: 'Unknown',
        24: 'Unknown',
        25: 'Transfer_other',
        26: 'Transfer_other'
    }

    df['discharge_disposition_id'] = df['discharge_disposition_id'].map(discharge_map).fillna('Unknown')
    df['admission_type_id']        = df['admission_type_id'].map(admission_type_map).fillna('Unknown')
    df['admission_source_id']      = df['admission_source_id'].map(admission_source_map).fillna('Unknown')

    # ── 4. Encode age brackets as ordered numbers ────────────────────────────
    # Age is stored as [0-10), [10-20) etc — convert to 0-9 scale
    age_map = {
        '[0-10)':   0, '[10-20)':  1, '[20-30)':  2,
        '[30-40)':  3, '[40-50)':  4, '[50-60)':  5,
        '[60-70)':  6, '[70-80)':  7, '[80-90)':  8,
        '[90-100)': 9
    }
    df['age'] = df['age'].map(age_map)

    # ── 5. Binary encode yes/no columns ─────────────────────────────────────
    binary_cols = ['change', 'diabetesMed']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # ── 6. Encode medication columns ─────────────────────────────────────────
    # Values are: No, Steady, Up, Down
    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col in med_cols:
        df[col] = df[col].map(med_map).fillna(0).astype(int)

    # ── 7. Label encode remaining categoricals ───────────────────────────────
    # race and medical_specialty have too many values for one-hot encoding
    le = LabelEncoder()
    for col in ['race', 'medical_specialty', 
            'discharge_disposition_id',
            'admission_type_id',
            'admission_source_id',
            'diag_1_cat', 'diag_2_cat', 'diag_3_cat']:
        df[col] = le.fit_transform(df[col].astype(str))
        

    # ── 8. Create a few engineered features ──────────────────────────────────
    # Total prior utilization (combines inpatient + emergency + outpatient)
    df['total_prior_visits'] = (
        df['number_inpatient'] +
        df['number_emergency'] +
        df['number_outpatient']
    )

    # Medication intensity (how many meds were actively changed)
    df['meds_changed'] = (df[med_cols] > 1).sum(axis=1)

    # ── 9. Separate features and target ─────────────────────────────────────
    X = df.drop(columns=['readmitted_30day'])
    y = df['readmitted_30day']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    print(f"\nFeature columns:\n{list(X.columns)}")

    return X, y


if __name__ == "__main__":
    X, y = build_features()

