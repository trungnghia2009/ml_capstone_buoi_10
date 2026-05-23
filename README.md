# Skin Cancer Detection — SVM (HAM10000 Metadata)

Phân loại tổn thương da từ metadata bệnh nhân (`age`, `sex`, `localization`) bằng SVM RBF. Hai task: binary (benign vs malignant) và multi-class (7 dx).

> Chỉ dùng metadata CSV, không xử lý ảnh dermoscopy → trần hiệu năng hạn chế.

---

## Cấu trúc

```
ml_capstone-buoi-10/
├── app/app.py                 # Streamlit UI
├── data/HAM10000_metadata.csv # 10015 rows
├── model/                     # *.joblib output
├── prepare_data/preprocess.py # Helpers
├── SVM/
│   ├── SVM_binary_class.ipynb
│   └── SVM_multi_class.ipynb
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu: Python 3.9+.

---

## Pipeline

1. `load_data` → `filter_valid_dx` (giữ 7 lớp chuẩn).
2. Label: `make_binary_label` (0/1) hoặc `make_multiclass_label` (LabelEncoder).
3. `group_split(group_col='lesion_id')` — GroupShuffleSplit 80/20, đảm bảo cùng lesion không cross train/test.
4. `build_full_pipeline()` — ColumnTransformer (impute + scale `age`, OHE `sex`/`localization`) → SVC RBF `class_weight='balanced'`. Encoder fit trong CV fold → không leak.
5. `grid_search_tune(scoring=..., n_splits=5)` — StratifiedGroupKFold 5-fold:
   - Binary: `scoring='average_precision'` (PR-AUC).
   - Multi: `scoring='f1_macro'`.
   - Grid: C ∈ [0.1, 1, 10, 100] × γ ∈ ['scale', 0.001, 0.01, 0.1] = 16 combos × 5 folds = 80 fits.
6. Binary thêm `tune_threshold(beta=2.0)` — F2 ưu tiên recall cho tầm soát.
7. Save bundle (`pipeline + raw_input=True + threshold/le_dx`).

**Tại sao `class_weight='balanced'` thay vì SMOTE/RUS?** Stack hai chiến lược → cân bằng kép. Phiên bản cũ RUS sụp `nv` 5364→92, model underfit. Một strategy là đủ.

**Tại sao group split theo `lesion_id`?** HAM10000 ~10000 ảnh nhưng ~7000 lesion. Row-level split rò rỉ lesion qua train/test → CV lạc quan giả.

---

## Chạy

```bash
# Notebook
cd SVM && jupyter lab

# App (sau khi 2 .joblib sinh)
streamlit run app/app.py
```

GridSearchCV ~3-7 phút (CPU, `n_jobs=-1`).

---

## Kết quả tham khảo

**Binary**
| metric | value |
|---|---|
| CV PR-AUC | 0.45 |
| Test PR-AUC | 0.42 |
| Test f1_macro (default thr) | 0.66 |
| Test f1_macro (F2-tuned thr) | 0.61 (recall malig 0.74→0.82) |
| CV-test gap | 0.03 |

**Multi**
| metric | value |
|---|---|
| CV f1_macro | 0.20 |
| Test f1_macro | 0.22 |
| Test balanced_acc | 0.35 |
| CV-test gap | 0.01 |

Trần thấp do metadata-only. Vượt trần cần image features.

---

## Dataset

`data/HAM10000_metadata.csv` — 10015 rows, ~7000 lesion.

| Column | Vai trò |
|---|---|
| `lesion_id` | group key cho split |
| `dx` | target (7 lớp) |
| `age`, `sex`, `localization` | features |
| còn lại | drop |

Class imbalance: `nv` 67%, `df` 1.1%. Missing: `age` 57 rows (impute median).

**Binary mapping**: malignant = {`mel, bcc, akiec, vasc`}, benign = {`nv, df, bkl`}.

Nguồn: [Tschandl et al. 2018](https://www.nature.com/articles/sdata2018161).

---

## Troubleshooting

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: preprocess` | `cd SVM` trước khi chạy |
| `KeyError: 'lesion_id'` | Giữ `lesion_id` đến sau group split |
| Streamlit "Model files not found" | Chạy cả 2 notebook trước |
| App predict crash bundle cũ | Chạy lại notebook để overwrite `.joblib` |

---

## Tech stack

`scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `streamlit` · `jupyter`
