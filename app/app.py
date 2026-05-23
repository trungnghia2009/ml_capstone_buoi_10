import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prepare_data')))

import streamlit as st
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
BINARY_PATH = os.path.join(MODEL_DIR, 'svm_binary_class_model.joblib')
MULTI_PATH  = os.path.join(MODEL_DIR, 'svm_multi_class_model.joblib')

# Fixed option lists (HAM10000 schema) — used when bundle has no fitted encoders.
SEX_FALLBACK = ['female', 'male', 'unknown']
LOC_FALLBACK = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot',
                'genital', 'hand', 'lower extremity', 'neck', 'scalp',
                'trunk', 'unknown', 'upper extremity']

st.set_page_config(page_title='Skin Cancer Detection', layout='centered')
st.title('Skin Cancer Detection — SVM (Metadata)')
st.caption('HAM10000 dataset · Features: age, sex, localization')


@st.cache_resource
def load_models():
    bin_bundle   = joblib.load(BINARY_PATH)
    multi_bundle = joblib.load(MULTI_PATH)
    return bin_bundle, multi_bundle


if not os.path.exists(BINARY_PATH) or not os.path.exists(MULTI_PATH):
    st.error('Model files not found. Run both notebooks first to generate .joblib files.')
    st.stop()

bin_bundle, multi_bundle = load_models()


def get_options(bundle):
    """Pull SEX/LOC option lists from bundle if present, else fallback."""
    if bundle.get('raw_input'):
        return SEX_FALLBACK, LOC_FALLBACK
    return list(bundle['le_sex'].classes_), list(bundle['ohe_loc'].categories_[0])


def encode_input(bundle, age, sex, localization):
    """Return model input. Raw-input bundles → DataFrame; pre-encoded bundles → np.ndarray."""
    if bundle.get('raw_input'):
        return pd.DataFrame([{'age': float(age), 'sex': sex, 'localization': localization}])
    sex_enc = bundle['le_sex'].transform([sex])[0]
    loc_arr = bundle['ohe_loc'].transform([[localization]])[0]
    return np.concatenate([[float(age), float(sex_enc)], loc_arr]).reshape(1, -1)


SEX_OPTIONS, LOC_OPTIONS = get_options(bin_bundle)

st.subheader('Patient Information')
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=45, step=1)
with col2:
    sex = st.selectbox('Sex', SEX_OPTIONS)
with col3:
    localization = st.selectbox('Lesion Localization', LOC_OPTIONS)

if st.button('Predict', type='primary'):
    st.divider()
    st.subheader('Results')

    col_b, col_m = st.columns(2)

    with col_b:
        st.markdown('**Binary Classification**')
        X_bin = encode_input(bin_bundle, age, sex, localization)
        score = bin_bundle['pipeline'].decision_function(X_bin)[0]
        thr   = bin_bundle.get('threshold', 0.0)
        pred  = int(score >= thr)
        label = bin_bundle['label_map'][pred]
        color = 'red' if pred == 1 else 'green'
        st.markdown(f':{color}[**{label}**]')
        st.caption(f'Decision score: {score:.3f}  (threshold: {thr:.3f})')

    with col_m:
        st.markdown('**Multi-Class Classification**')
        X_multi = encode_input(multi_bundle, age, sex, localization)
        pred_multi = multi_bundle['pipeline'].predict(X_multi)[0]
        label_multi = multi_bundle['le_dx'].inverse_transform([pred_multi])[0]
        st.markdown(f'**{label_multi.upper()}**')
        scores = multi_bundle['pipeline'].decision_function(X_multi)[0]
        classes = multi_bundle['le_dx'].classes_
        score_df = pd.DataFrame({'class': classes, 'score': scores}).sort_values('score', ascending=False)
        st.dataframe(score_df.reset_index(drop=True), use_container_width=True)
