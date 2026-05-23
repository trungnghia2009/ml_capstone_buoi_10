"""Shared preprocessing helpers for HAM10000 SVM pipeline.

Used by both `SVM_binary_class.ipynb` and `SVM_multi_class.ipynb`.
Pipeline order: load → filter → label → group_split → build_full_pipeline → grid_search_tune.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# 7 standard HAM10000 dx classes. Any row with dx not in this set is dropped.
VALID_DX = {'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'}

# Binary label mapping per project spec (PDF):
#   1 = malignant/precancer, 0 = benign
BINARY_MALIGNANT = {'mel', 'bcc', 'akiec', 'vasc'}
BINARY_BENIGN    = {'nv', 'df', 'bkl'}

DX_BINARY_MAP = {dx: 1 for dx in BINARY_MALIGNANT}
DX_BINARY_MAP.update({dx: 0 for dx in BINARY_BENIGN})

RAW_FEATURES = ['age', 'sex', 'localization']


def load_data(csv_path: str) -> pd.DataFrame:
    """Load HAM10000 metadata CSV into DataFrame."""
    return pd.read_csv(csv_path)


def filter_valid_dx(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with dx not in VALID_DX."""
    return df[df['dx'].isin(VALID_DX)].reset_index(drop=True)


def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add `diagnosis` col mapping dx → 0/1 via DX_BINARY_MAP."""
    df = df.copy()
    df['diagnosis'] = df['dx'].map(DX_BINARY_MAP)
    return df


def make_multiclass_label(df: pd.DataFrame):
    """Add `diagnosis` col with LabelEncoder over 7 dx classes.

    Returns (df, fitted LabelEncoder) — encoder needed at inference to map int → dx name.
    """
    df = df.copy()
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['dx'])
    return df, le


def build_full_pipeline(class_weight='balanced') -> Pipeline:
    """Raw-input SVM pipeline. Imputer + scaler/OHE fit inside CV folds (no leak).

    Input: DataFrame with cols ['age', 'sex', 'localization'].
    Steps:
      - age:          median impute → StandardScaler
      - sex:          most_frequent impute → OneHotEncoder
      - localization: OneHotEncoder (handle_unknown='ignore')
      - SVC RBF with class_weight (single imbalance strategy).
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
    """GroupShuffleSplit by lesion_id — same lesion never crosses train/test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def grid_search_tune(X_train, y_train, groups, scoring: str = 'average_precision',
                     n_splits: int = 5):
    """GridSearchCV with StratifiedGroupKFold (no lesion leak across folds).

    Default scoring=`average_precision` (PR-AUC, robust on imbalanced binary).
    For multi-class, pass scoring='f1_macro'.
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
