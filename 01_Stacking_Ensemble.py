# stacking_app.py
# Streamlit: Stacking Ensemble Learning from scratch (manual OOF stacking)
# Run: streamlit run stacking_app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from copy import deepcopy
from typing import Dict, List, Tuple

from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import plotly.express as px
import plotly.graph_objects as go


# ----------------------------- Utilities -----------------------------
def set_page():
    st.set_page_config(page_title="Stacking Ensemble from Scratch", layout="wide")
    st.title("Stacking Ensemble Learning — from scratch")
    st.caption("Manual out-of-fold (OOF) stacking with scikit-learn + Streamlit")


def explain_basics():
    with st.expander("What is stacking? (Beginner friendly) 🧠", expanded=True):
        st.write(
            "- We train multiple base models (e.g., Logistic Regression, KNN, Decision Tree...).\n"
            "- Using cross-validation on the training set, we create out-of-fold (OOF) predictions from each base model.\n"
            "- These OOF predictions become new features for a meta-learner (often Logistic Regression).\n"
            "- At inference, we average each base model’s test predictions across folds and feed them to the meta-learner to get final predictions.\n\n"
            "Why OOF? If we trained the meta-learner on predictions from models evaluated on the same data they trained on, "
            "the meta-learner would see overly optimistic (leaky) signals. OOF ensures the meta-learner only sees predictions "
            "for samples that were not used to fit the base model that produced them."
        )


def load_builtin(name: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if name == "Iris":
        data = load_iris(as_frame=True)
    elif name == "Wine":
        data = load_wine(as_frame=True)
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
    else:
        raise ValueError("Unknown dataset")
    df = data.frame.copy()
    X = df.drop(columns=[data.target.name])
    y = df[data.target.name]
    return X, y, list(X.columns)


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])

    # Version-safe OneHotEncoder
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def align_proba_columns(proba: np.ndarray, model_classes: np.ndarray, all_classes: List) -> np.ndarray:
    """Map model's proba columns to a fixed class order."""
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    col_index = {c: i for i, c in enumerate(all_classes)}
    for i, c in enumerate(model_classes):
        out[:, col_index[c]] = proba[:, i]
    return out


def safe_predict_proba(estimator: Pipeline, X: pd.DataFrame, all_classes: List) -> np.ndarray:
    """Always return probability-like outputs aligned to all_classes."""
    model = estimator.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = estimator.predict_proba(X)
        return align_proba_columns(proba, model.classes_, all_classes)
    # Fallback: use decision_function and convert to probabilities
    if hasattr(model, "decision_function"):
        df = estimator.decision_function(X)
        if df.ndim == 1:
            # binary case -> sigmoid
            p1 = 1 / (1 + np.exp(-df))
            proba = np.vstack([1 - p1, p1]).T
            # assume classes_ has two entries
            return align_proba_columns(proba, model.classes_, all_classes)
        else:
            # multi-class -> softmax
            ex = np.exp(df - df.max(axis=1, keepdims=True))
            proba = ex / ex.sum(axis=1, keepdims=True)
            return align_proba_columns(proba, model.classes_, all_classes)
    # Last resort: hard labels to pseudo-proba (one-hot)
    labels = estimator.predict(X)
    proba = np.zeros((len(labels), len(all_classes)))
    for i, lab in enumerate(labels):
        proba[i, all_classes.index(lab)] = 1.0
    return proba


def make_base_models(params) -> List[Tuple[str, object]]:
    """Create selected base models."""
    models = []
    if params["use_lr"]:
        models.append(("LogReg", LogisticRegression(max_iter=200, C=params["lr_C"], n_jobs=None, solver="lbfgs")))
    if params["use_knn"]:
        models.append(("KNN", KNeighborsClassifier(n_neighbors=params["knn_k"])))
    if params["use_dt"]:
        models.append(("DecisionTree", DecisionTreeClassifier(max_depth=params["dt_depth"] or None, random_state=params["seed"])))
    if params["use_rf"]:
        models.append(("RandomForest", RandomForestClassifier(
            n_estimators=params["rf_estimators"], max_depth=params["rf_depth"] or None,
            random_state=params["seed"], n_jobs=-1)))
    if params["use_svc"]:
        models.append(("SVC", SVC(C=params["svc_C"], kernel="rbf", probability=True, random_state=params["seed"])))
    return models


def stacking_oof(
    base_models: List[Tuple[str, object]],
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,
    all_classes: List,
    cv_splits: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Manual OOF stacking:
    - Returns OOF features for train (Z_train), averaged test features (Z_test),
      feature names, plus per-model test proba and per-model OOF (for diagnostics).
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_classes = len(all_classes)

    Z_train_parts = []  # list of (n_train, n_classes) per model
    Z_test_parts = []   # list of (n_test, n_classes) per model
    feat_names = []
    per_model_test = {}
    per_model_oof = {}

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for name, base in base_models:
        # Out-of-fold holder for this model
        oof = np.zeros((n_train, n_classes), dtype=float)
        test_accum = np.zeros((n_test, n_classes), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            # Build a fresh preprocess + model for each fold to avoid leakage
            pre = build_preprocess(X_tr)
            model = clone(base)
            pipe = Pipeline([("preprocess", pre), ("model", model)])
            pipe.fit(X_tr, y_tr)

            # Validate fold predictions
            va_proba = safe_predict_proba(pipe, X_va, all_classes)
            oof[va_idx, :] = va_proba

            # Test predictions (accumulate for averaging)
            te_proba = safe_predict_proba(pipe, X_test, all_classes)
            test_accum += te_proba

        # Average test predictions across folds
        test_mean = test_accum / cv_splits

        Z_train_parts.append(oof)
        Z_test_parts.append(test_mean)
        per_model_test[name] = test_mean
        per_model_oof[name] = oof

        # Feature names for this model's class-prob columns
        feat_names.extend([f"{name}_p({c})" for c in all_classes])

    # Concatenate across models
    Z_train = np.hstack(Z_train_parts) if Z_train_parts else np.zeros((n_train, 0))
    Z_test = np.hstack(Z_test_parts) if Z_test_parts else np.zeros((n_test, 0))

    return Z_train, Z_test, feat_names, per_model_test, per_model_oof


def metric_dict(y_true, y_pred, y_proba, classes: List) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }
    try:
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    except Exception:
        out["log_loss"] = np.nan

    # ROC-AUC (binary -> standard; multiclass -> OVR macro)
    try:
        if len(classes) == 2:
            # use proba for positive class
            pos_idx = list(classes).index(classes[-1])
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except Exception:
        out["roc_auc"] = np.nan
    return out


def plot_metric_bars(scores: Dict[str, Dict[str, float]], metric: str):
    rows = []
    for name, d in scores.items():
        rows.append({"Model": name, metric: d.get(metric, np.nan)})
    df = pd.DataFrame(rows).sort_values(metric, ascending=False)
    fig = px.bar(df, x="Model", y=metric, title=f"{metric} by model")
    fig.update_layout(xaxis_tickangle=-15)
    return fig


def plot_confusion(y_true, y_pred, labels, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title=title
    )
    return fig


def plot_roc_curves(prob_map: Dict[str, np.ndarray], y_true: pd.Series, classes: List):
    fig = go.Figure()
    if len(classes) == 2:
        pos_label = classes[-1]
        pos_idx = list(classes).index(pos_label)
        for name, proba in prob_map.items():
            fpr, tpr, _ = roc_curve(y_true == pos_label, proba[:, pos_idx])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc(fpr,tpr):.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC curves (binary)", xaxis_title="FPR", yaxis_title="TPR")
    else:
        # Micro-average ROC
        # Stack one-vs-rest
        y_bin = pd.get_dummies(pd.Categorical(y_true, categories=classes)).values
        for name, proba in prob_map.items():
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (micro AUC={auc(fpr,tpr):.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC curves (multiclass, micro-average)", xaxis_title="FPR", yaxis_title="TPR")
    return fig


def plot_meta_coefficients(model: LogisticRegression, feat_names: List[str], classes: List):
    fig = go.Figure()
    if len(classes) == 2:
        # binary: single coef vector (shape [n_features])
        coefs = model.coef_.reshape(-1)
        df = pd.DataFrame({"feature": feat_names, "coef_abs": np.abs(coefs), "coef": coefs})
        df = df.sort_values("coef_abs", ascending=False).head(20).sort_values("coef_abs")
        fig = px.bar(df, x="coef_abs", y="feature", orientation="h", title="Meta-learner | top abs coefficients")
    else:
        # multiclass: sum absolute across classes
        coefs = model.coef_
        agg = np.sum(np.abs(coefs), axis=0)
        df = pd.DataFrame({"feature": feat_names, "coef_abs_sum": agg})
        df = df.sort_values("coef_abs_sum", ascending=False).head(20).sort_values("coef_abs_sum")
        fig = px.bar(df, x="coef_abs_sum", y="feature", orientation="h", title="Meta-learner | top abs coefficients (summed over classes)")
    return fig


# ----------------------------- App UI -----------------------------
def main():
    set_page()
    explain_basics()

    with st.sidebar:
        st.header("Data")
        data_choice = st.selectbox("Choose dataset", ["Iris", "Wine", "Breast Cancer", "Upload CSV"])
        df = None
        target_col = None
        if data_choice == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded)
                st.write(f"Loaded data shape: {df.shape}")
                target_col = st.selectbox("Target column", df.columns)
        else:
            uploaded = None

        st.header("Train/Test + CV")
        test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2, step=0.05)
        cv_splits = st.slider("OOF CV folds", 3, 10, 5, step=1)
        seed = st.number_input("Random state", 0, 10000, 42, step=1)

        st.header("Base models")
        use_lr = st.checkbox("Logistic Regression", True)
        lr_C = st.slider("LR: C", 0.01, 10.0, 1.0, step=0.01)

        use_knn = st.checkbox("K-Nearest Neighbors", True)
        knn_k = st.slider("KNN: n_neighbors", 1, 25, 5, step=1)

        use_dt = st.checkbox("Decision Tree", True)
        dt_depth_toggle = st.radio("DT: max_depth", ["None", "Set"], horizontal=True, index=0)
        dt_depth = st.slider("DT: depth value", 1, 50, 6) if dt_depth_toggle == "Set" else None

        use_rf = st.checkbox("Random Forest", True)
        rf_estimators = st.slider("RF: n_estimators", 50, 500, 200, step=50)
        rf_depth_toggle = st.radio("RF: max_depth", ["None", "Set"], horizontal=True, index=0)
        rf_depth = st.slider("RF: depth value", 2, 50, 12) if rf_depth_toggle == "Set" else None

        use_svc = st.checkbox("Support Vector Classifier (probability=True)", True)
        svc_C = st.slider("SVC: C", 0.01, 10.0, 1.0, step=0.01)

        st.header("Run")
        metric_to_plot = st.selectbox("Metric to plot", ["accuracy", "f1_macro", "roc_auc", "log_loss"], index=0)
        run_btn = st.button("Run experiment")

    # -------------------- Load data --------------------
    if data_choice == "Upload CSV":
        if uploaded is None or target_col is None:
            st.info("Upload a CSV and choose a target to continue.")
            st.stop()
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X, y, feature_cols = load_builtin(data_choice)

    # basic checks
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 30:
        st.warning("Target has many unique numeric values; this looks like regression. Please use a classification target.")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(seed), stratify=y
    )
    classes = list(pd.Series(y_train).unique())

    st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Classes: {classes}")

    if not run_btn:
        st.info("Configure options in the sidebar, then click 'Run experiment'.")
        st.stop()

    # -------------------- Build models --------------------
    params = dict(
        use_lr=use_lr, lr_C=float(lr_C),
        use_knn=use_knn, knn_k=int(knn_k),
        use_dt=use_dt, dt_depth=None if dt_depth_toggle == "None" else int(dt_depth),
        use_rf=use_rf, rf_estimators=int(rf_estimators), rf_depth=None if rf_depth_toggle == "None" else int(rf_depth),
        use_svc=use_svc, svc_C=float(svc_C),
        seed=int(seed)
    )
    base_models = make_base_models(params)
    if not base_models:
        st.error("Please select at least one base model.")
        st.stop()

    # -------------------- Fit base learners + OOF stacking --------------------
    with st.spinner("Computing OOF predictions for stacking..."):
        Z_tr, Z_te, feat_names, per_model_test, per_model_oof = stacking_oof(
            base_models, X_train, y_train, X_test, classes, cv_splits=cv_splits, seed=seed
        )

    st.success(f"Built stacking features: train {Z_tr.shape}, test {Z_te.shape}")
    st.caption("Each base model contributes one probability column per class.")

    # -------------------- Meta-learner --------------------
    st.subheader("Meta-learner (Level-2)")
    st.write("We use Logistic Regression on OOF features.")
    meta = LogisticRegression(max_iter=400, solver="lbfgs", n_jobs=None)
    meta.fit(Z_tr, y_train)
    proba_stack_test = meta.predict_proba(Z_te)
    pred_stack_test = meta.predict(Z_te)

    # -------------------- Evaluate base vs stacked --------------------
    st.subheader("Evaluation on Test Set")
    metric_map: Dict[str, Dict[str, float]] = {}

    # Evaluate base models (trained on full train with preprocessing)
    with st.spinner("Training base models on full training set for test evaluation..."):
        for name, base in base_models:
            pre = build_preprocess(X_train)
            pipe = Pipeline([("preprocess", pre), ("model", clone(base))])
            pipe.fit(X_train, y_train)
            proba = safe_predict_proba(pipe, X_test, classes)
            preds = pipe.predict(X_test)
            metric_map[name] = metric_dict(y_test, preds, proba, classes)

    # Stacked
    metric_map["STACK"] = metric_dict(y_test, pred_stack_test, proba_stack_test, classes)

    # Metrics table
    df_metrics = pd.DataFrame(metric_map).T[["accuracy", "f1_macro", "roc_auc", "log_loss"]]
    st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)

    # Bar plot of the chosen metric
    fig_bar = plot_metric_bars(metric_map, metric_to_plot)
    st.plotly_chart(fig_bar, width="stretch")

    # Confusion matrix (stacked)
    st.subheader("Confusion matrix — STACK")
    fig_cm = plot_confusion(y_test, pred_stack_test, labels=classes, title="STACK confusion matrix (test)")
    st.plotly_chart(fig_cm, width="stretch")
    st.text("Classification report (STACK):")
    st.text(classification_report(y_test, pred_stack_test, zero_division=0))

    # ROC curves
    st.subheader("ROC curves")
    prob_for_roc = {**per_model_test, "STACK": proba_stack_test}
    fig_roc = plot_roc_curves(prob_for_roc, y_test, classes)
    st.plotly_chart(fig_roc, width="stretch")

    # Meta-learner coefficients
    st.subheader("Meta-learner feature importance (coefficients)")
    fig_coef = plot_meta_coefficients(meta, feat_names, classes)
    st.plotly_chart(fig_coef, width="stretch")

    # Correlation of base model predictions (binary: positive class only)
    if len(classes) == 2:
        st.subheader("Correlation of base model predicted probabilities (positive class)")
        pos_idx = list(classes).index(classes[-1])
        corr_df = pd.DataFrame({name: proba[:, pos_idx] for name, proba in per_model_test.items()})
        corr = corr_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                             title="Correlation heatmap (test proba)")
        st.plotly_chart(fig_corr, width="stretch")

    with st.expander("How to read these results"):
        st.write(
            "- The table shows test metrics for each base model and the STACK meta-model.\n"
            "- ROC curves compare ranking quality (AUC). In multiclass we show the micro-average (flattened one-vs-rest).\n"
            "- The meta-learner coefficients indicate how strongly each base model’s class-probability feature influences the final decision.\n"
            "- If STACK outperforms all base models, the meta-learner successfully captured complementary strengths."
        )


if __name__ == "__main__":
    main()