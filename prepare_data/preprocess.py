"""Shared preprocessing helpers for HAM10000 SVM pipeline.

Used by both `SVM_binary_class.ipynb` and `SVM_multi_class.ipynb`.
Pipeline order: load → filter → impute → label → encode → split → tune.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# 7 standard HAM10000 dx classes. Any row with dx not in this set is dropped.
VALID_DX = {'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}

# Binary label mapping per project spec (PDF):
#   B=1 malignant/precancer, M=0 benign
BINARY_MALIGNANT = {'mel', 'bcc', 'akiec', 'vasc'}
BINARY_BENIGN    = {'nv', 'df', 'bkl'}

DX_BINARY_MAP = {dx: 1 for dx in BINARY_MALIGNANT}
DX_BINARY_MAP.update({dx: 0 for dx in BINARY_BENIGN})

# Metadata cols irrelevant for classification — drop after loading.
DROP_COLS = ['lesion_id', 'image_id', 'dataset', 'dx_type']


def load_data(csv_path: str) -> pd.DataFrame:
    """Load HAM10000 metadata CSV into DataFrame."""
    return pd.read_csv(csv_path)


def filter_valid_dx(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with dx not in VALID_DX (e.g. removes non-standard 'healthy')."""
    return df[df['dx'].isin(VALID_DX)].reset_index(drop=True)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for `age`, most_frequent for `sex`."""
    df = df.copy()
    age_imp = SimpleImputer(strategy='median')
    df['age'] = age_imp.fit_transform(df[['age']])
    sex_imp = SimpleImputer(strategy='most_frequent')
    df['sex'] = sex_imp.fit_transform(df[['sex']]).ravel()
    return df


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add `diagnosis` col mapping dx → 0/1 via DX_BINARY_MAP (binary task)."""
    df = df.copy()
    df['diagnosis'] = df['dx'].map(DX_BINARY_MAP)
    return df


def make_multiclass_label(df: pd.DataFrame):
    """Add `diagnosis` col with LabelEncoder over 7 dx classes (multi-class task).

    Returns (df, fitted LabelEncoder) — encoder needed at inference to map int → dx name.
    """
    df = df.copy()
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['dx'])
    return df, le


def encode_features(df: pd.DataFrame):
    """Encode categorical features for SVM:
      - `sex` → LabelEncoder (binary 0/1)
      - `localization` → OneHotEncoder (15 cols, nominal — avoid ordinal assumption)

    Returns (df, le_sex, ohe_loc, loc_cols). Encoders are returned for reuse
    during inference (Streamlit app encodes user input the same way).
    """
    df = df.copy()
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])

    ohe_loc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    loc_arr = ohe_loc.fit_transform(df[['localization']])
    loc_cols = [f'loc_{c}' for c in ohe_loc.categories_[0]]
    loc_df = pd.DataFrame(loc_arr, columns=loc_cols, index=df.index)

    df = df.drop(columns=['localization'])
    df = pd.concat([df, loc_df], axis=1)
    return df, le_sex, ohe_loc, loc_cols


def split_data(df: pd.DataFrame, feature_cols: list, target_col: str = 'diagnosis'):
    """Stratified 80/20 train/test split. Drops any rows with NaN in feature/target."""
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].values
    y = df[target_col].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_svm_pipeline(n_continuous: int = 1) -> ImbPipeline:
    """Build imblearn Pipeline: scale → undersample → SVC.

    Steps:
      1. ColumnTransformer — StandardScaler on first `n_continuous` cols (e.g. `age`),
         passthrough rest (already-encoded sex + one-hot loc).
      2. RandomUnderSampler — downsample majority class (`nv`) to next-largest count.
         Only active during fit; skipped at predict time (imblearn behavior).
      3. SVC — RBF kernel, class_weight='balanced' for residual imbalance,
         cache_size=1000 MB for faster kernel computation.
    """
    continuous_idx = list(range(n_continuous))
    preproc = ColumnTransformer(
        [('scaler', StandardScaler(), continuous_idx)],
        remainder='passthrough',
    )
    return ImbPipeline([
        ('preproc', preproc),
        ('rus',     RandomUnderSampler(sampling_strategy='majority', random_state=42)),
        ('svc',     SVC(class_weight='balanced', cache_size=1000, random_state=42)),
    ])


# --- v2: raw-input pipeline, group-aware split, threshold-friendly tuning ---

RAW_FEATURES = ['age', 'sex', 'localization']


def build_full_pipeline(class_weight='balanced') -> Pipeline:
    """Raw-input SVM pipeline. Imputer + scaler/OHE fit inside CV folds (no leak).

    Input: DataFrame with cols ['age', 'sex', 'localization'].
    Steps:
      - age:          median impute → StandardScaler
      - sex:          most_frequent impute → OneHotEncoder
      - localization: OneHotEncoder (handle_unknown='ignore')
      - SVC RBF with class_weight (single imbalance strategy; no undersampler).
    """
    age_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
    ])
    sex_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    loc_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preproc = ColumnTransformer([
        ('age', age_pipe, ['age']),
        ('sex', sex_pipe, ['sex']),
        ('loc', loc_pipe, ['localization']),
    ])
    return Pipeline([
        ('preproc', preproc),
        ('svc',     SVC(class_weight=class_weight, cache_size=1000,
                        probability=False, random_state=42)),
    ])


def group_split(df: pd.DataFrame, target_col: str = 'diagnosis',
                group_col: str = 'lesion_id', test_size: float = 0.2,
                random_state: int = 42):
    """GroupShuffleSplit by lesion_id — same lesion never crosses train/test.

    Returns (df_train, df_test). Caller selects feature cols downstream.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def grid_search_tune_v2(X_train, y_train, groups, scoring: str = 'average_precision',
                        n_splits: int = 5):
    """GridSearchCV with StratifiedGroupKFold (no lesion leak across folds).

    Wider grid than v1. Default scoring=`average_precision` (PR-AUC, robust
    on imbalanced binary). For multi-class, pass scoring='f1_macro'.
    """
    pipe = build_full_pipeline()
    param_grid = {
        'svc__C':      [0.1, 1, 10, 100],
        'svc__gamma':  ['scale', 0.001, 0.01, 0.1],
        'svc__kernel': ['rbf'],
    }
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=True, verbose=1,
    )
    gs.fit(X_train, y_train, groups=groups)
    return gs


def tune_threshold(y_true, scores, beta: float = 1.0):
    """Pick threshold on decision_function scores that maximizes F-beta.

    beta=1 → F1. beta>1 favors recall (clinical screening).
    Returns (best_threshold, best_fbeta).
    """
    from sklearn.metrics import precision_recall_curve, fbeta_score
    _, _, thr = precision_recall_curve(y_true, scores)
    best_t, best_f = 0.0, -1.0
    for t in thr:
        y_hat = (scores >= t).astype(int)
        f = fbeta_score(y_true, y_hat, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def grid_search_tune(X_train, y_train, scoring: str = 'f1_macro', n_continuous: int = 1):
    """Hyperparameter tuning via 3-fold StratifiedKFold GridSearchCV.

    Search space (9 combos × 3 folds = 27 fits):
      - C:      [1, 10, 100]
      - gamma:  ['scale', 0.01, 0.1]
      - kernel: ['rbf']

    Wraps `build_svm_pipeline` so RandomUnderSampler runs inside each CV fold
    (no leak into validation fold). verbose=2 prints per-fit progress.

    Returns fitted GridSearchCV. Access best pipeline via `.best_estimator_`.
    """
    pipe = build_svm_pipeline(n_continuous=n_continuous)
    param_grid = {
        'svc__C':      [1, 10, 100],
        'svc__gamma':  ['scale', 0.01, 0.1],
        'svc__kernel': ['rbf'],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        verbose=2,
    )
    gs.fit(X_train, y_train)
    return gs
