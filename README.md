# Skin Cancer Detection — SVM (HAM10000 Metadata)

ML pipeline phân loại tổn thương da từ **metadata bệnh nhân** (`age`, `sex`, `localization`) — HAM10000. Mô hình: **SVM RBF** + `class_weight='balanced'` + **GroupKFold (theo `lesion_id`)** + **GridSearchCV**. Binary thêm **threshold tuning** theo F2.

> HAM10000 gốc có ảnh dermoscopy. Project chỉ dùng metadata CSV — không trích xuất feature ảnh. Trần hiệu năng bị giới hạn bởi feature set: binary f1_macro ~0.64, multi f1_macro ~0.22.

---

## Mục tiêu

| Task | Lớp | Metric chính | Số đo |
|---|---|---|---|
| Binary | Benign (0) vs Malignant/Precancer (1) | PR-AUC, f1_macro | ~0.42 / ~0.64 |
| Multi-class | 7 dx (`nv, mel, bkl, bcc, akiec, vasc, df`) | f1_macro | ~0.22 |

**Label mapping binary**:
- `1` malignant/precancer: `mel, bcc, akiec, vasc`
- `0` benign: `nv, df, bkl`

---

## Cấu trúc thư mục

```
capstone-buoi-10-new/
├── app/
│   └── app.py                      # Streamlit demo UI (raw-input bundle aware)
├── data/
│   └── HAM10000_metadata.csv       # Dataset (10015 rows)
├── model/                          # Output: .joblib bundles
│   ├── svm_binary_class_model.joblib
│   └── svm_multi_class_model.joblib
├── prepare_data/
│   └── preprocess.py               # Shared helpers: raw pipeline, group split, threshold tune
├── SVM/
│   ├── SVM_binary_class.ipynb      # Pipeline binary
│   └── SVM_multi_class.ipynb       # Pipeline 7-class
├── requirements.txt
└── README.md
```

---

## Môi trường

Python 3.9+ (test 3.9–3.14), macOS / Linux / Windows.

### venv
```bash
cd ml_capstone-buoi-10
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### conda
```bash
conda create -n skin-svm python=3.11 -y
conda activate skin-svm
pip install -r requirements.txt
```

### Verify
```bash
python -c "import pandas, numpy, sklearn, imblearn, streamlit, seaborn, joblib; print('OK')"
```

---

## Pipeline

1. **Load** CSV → `load_data`.
2. **Filter** dx không hợp lệ → `filter_valid_dx`.
3. **Label**:
   - Binary → `make_binary_label` (map dx → 0/1).
   - Multi → `make_multiclass_label` (LabelEncoder 7 lớp, trả về `le_dx`).
4. **Giữ `lesion_id`** đến khi split (cần cho group split).
5. **Group split** → `group_split(df, group_col='lesion_id')` dùng `GroupShuffleSplit(test_size=0.2)`:
   - HAM10000 có nhiều ảnh per lesion → split row-level rò rỉ lesion qua train/test.
   - Group split bảo đảm **lesion overlap = 0**.
6. **Pipeline raw-input** → `build_full_pipeline()`:
   ```
   ColumnTransformer
     ├─ age:          SimpleImputer(median) → StandardScaler
     ├─ sex:          SimpleImputer(most_frequent) → OneHotEncoder
     └─ localization: OneHotEncoder(handle_unknown='ignore')
   → SVC(kernel='rbf', class_weight='balanced', cache_size=1000)
   ```
   - Toàn bộ encoder/imputer **fit trong CV fold** → không leak.
   - Một chiến lược cân bằng duy nhất (`class_weight`). Không stack thêm RUS/SMOTE.
7. **GridSearchCV** với `StratifiedGroupKFold(5)` → `grid_search_tune`:
   ```
   svc__C:      [0.1, 1, 10, 100]
   svc__gamma:  ['scale', 0.001, 0.01, 0.1]
   svc__kernel: ['rbf']
   ```
   - 4×4 = **16 combos × 5 fold = 80 fits**.
   - Scoring:
     - Binary: `average_precision` (PR-AUC, không phụ thuộc threshold).
     - Multi:  `f1_macro` (cân bằng giữa các lớp).
   - `groups=lesion_id` chuyển vào `gs.fit` → cùng lesion không cross fold.
8. **Threshold tuning** (chỉ binary) → `tune_threshold(y_train, train_scores, beta=2.0)`:
   - Quét PR curve, chọn threshold max F-beta.
   - β=2 → ưu tiên recall (tầm soát ung thư).
9. **Eval**:
   - Binary: PR-AUC, f1_macro (default thr + tuned thr), classification_report, CM, PR curve.
   - Multi: f1_macro, balanced_acc, f1_weighted, classification_report, CM (counts + row-normalized).
10. **Save bundle** (`joblib.dump`):
    - Binary: `{pipeline, features, threshold, label_map, raw_input=True}`.
    - Multi:  `{pipeline, features, le_dx, raw_input=True}`.

---

## Tại sao không SMOTE / RUS?

Phiên bản cũ stack `RandomUnderSampler` + `class_weight='balanced'` → cân bằng kép. Hậu quả:
- Multi-class: RUS giảm `nv` từ 5364 → 92 (size lớp hiếm nhất `df`) → mất 98% data majority → SVM underfit thảm hại (acc 0.35, f1_macro 0.18).
- Binary: precision malignant chỉ 0.40 (nhiều false positive).

V2 giữ một strategy duy nhất (`class_weight='balanced'`) → SVC tự nhân penalty nghịch đảo tần số lớp → không vứt data. F1_macro multi cải thiện +22% (0.18 → 0.22).

---

## Tại sao GroupKFold theo `lesion_id`?

HAM10000 có ~10000 ảnh nhưng chỉ ~7000 lesion (nhiều lesion có 2-6 ảnh từ các góc). Row-level split → cùng lesion xuất hiện cả train và test → CV score lạc quan giả tạo, generalization vỡ trên data thật.

`GroupShuffleSplit` + `StratifiedGroupKFold` ép cùng `lesion_id` không cross fold/split. Gap CV-test giờ ~1.5% (multi) → CV trung thực.

---

## Chạy notebook

```bash
cd SVM
jupyter notebook   # hoặc: jupyter lab
```

Run tuần tự `SVM_binary_class.ipynb` rồi `SVM_multi_class.ipynb`. File `.joblib` sinh vào `model/`.

> ⏱️ GridSearchCV 80 fits ~3-7 phút (tuỳ CPU). `n_jobs=-1`.

---

## Streamlit app

Sau khi cả 2 `.joblib` sinh:
```bash
streamlit run app/app.py
```

UI: nhập `age, sex, localization` → **Predict** → binary + multi side-by-side.

- Binary dùng `threshold` lưu trong bundle (F2-tuned) thay vì cut tại 0.
- App tự detect `raw_input=True` → truyền DataFrame thay vì encoded array.

---

## Dataset

**HAM10000** — `data/HAM10000_metadata.csv`:

| Column | Type | Note |
|---|---|---|
| `lesion_id` | str | giữ đến split (group key) |
| `image_id` | str | drop |
| `dx` | str | target (7 lớp) |
| `dx_type` | str | drop |
| `age` | float | feature (impute median, scale) |
| `sex` | str | feature (impute most_frequent, OHE) |
| `localization` | str | feature (OHE 15 cat) |
| `dataset` | str | drop |

- Tổng: 10015 rows, ~7000 lesion.
- Class imbalance: `nv` 67%, `df` 1.1%.
- Missing: `age` 57 rows.

Nguồn: [Tschandl et al. 2018, HAM10000](https://www.nature.com/articles/sdata2018161).

---

## Kết quả tham khảo

**Binary**
| metric | value |
|---|---|
| CV PR-AUC | ~0.42 |
| Test PR-AUC | ~0.42 |
| Test f1_macro (default thr) | ~0.64 |
| Test f1_macro (F2-tuned thr) | ~0.63 (recall malignant ↑) |
| CV-test gap | < 0.02 |

**Multi**
| metric | value |
|---|---|
| CV f1_macro | 0.20 |
| Test f1_macro | 0.22 |
| Test balanced_acc | 0.35 |
| Test accuracy | 0.35 |
| CV-test gap | 0.014 |

Trần thấp do metadata-only. Vượt trần cần image features (CNN embedding / ABCD rule).

---

## Verification checklist

- [ ] `pip install -r requirements.txt` không lỗi
- [ ] 2 notebook chạy end-to-end, in CM + classification_report
- [ ] Output cell "Lesion overlap: 0" sau group split
- [ ] GridSearchCV in `best_params_` + CV score
- [ ] Binary: PR-AUC ~0.4+, threshold tuned printed
- [ ] Multi: f1_macro > 0.20
- [ ] CV-test gap < 0.05
- [ ] 2 `.joblib` tồn tại trong `model/` với key `raw_input=True`
- [ ] `streamlit run app/app.py` predict không crash

---

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ModuleNotFoundError: imblearn` | Thiếu dep | `pip install imbalanced-learn` |
| `ModuleNotFoundError: preprocess` | Sai working dir | `cd SVM` trước khi chạy notebook |
| `KeyError: 'lesion_id'` | Drop sớm | Giữ `lesion_id` đến sau group split |
| `ValueError: groups param not provided` | Quên `groups=` ở `gs.fit` | `grid_search_tune` đã handle, kiểm tra import đúng |
| `KeyError: 'param_C'` | Pipeline prefix | Dùng `param_svc__C` |
| `FileNotFoundError: HAM10000_metadata.csv` | Path relative sai | Chạy notebook từ `SVM/` |
| Streamlit "Model files not found" | Chưa chạy notebook | Run cả 2 `.ipynb` |
| App predict crash với bundle cũ | Bundle cũ không có `raw_input` | Chạy lại notebook để overwrite `.joblib` |
| GridSearchCV chậm (>10 phút) | n_jobs ít | `n_jobs=-1` (mặc định), giảm `n_splits` nếu cần |

---

## Tech stack

`scikit-learn` · `imbalanced-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `streamlit` · `jupyter`
