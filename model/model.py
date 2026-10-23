"""Training utilities for the Smart Outcome Predictor project."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor, RandomForestClassifier,
    RandomForestRegressor, VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CLASS = "completion_status"
TARGET_REG = "final_score"
DROP_COLUMNS = {"student_id", "course_start_date"}


def load_dataset(uploaded_file=None, default_path="data/Smart_Outcome_Predictor_Dataset_5200.xlsx"):
    """Load CSV or Excel data from Streamlit upload or a local project path."""
    source = uploaded_file if uploaded_file is not None else default_path
    if uploaded_file is None:
        import os
        if not os.path.exists(source):
            # Also support running main.py from a different working directory.
            source = os.path.join(os.path.dirname(__file__), "data", "Smart_Outcome_Predictor_Dataset_5200.xlsx")
        if not os.path.exists(source):
            raise FileNotFoundError("Dataset not found. Upload the provided Excel/CSV dataset in the app.")
    name = getattr(source, "name", str(source)).lower()
    if name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(source)
    elif name.endswith('.csv'):
        df = pd.read_csv(source)
    else:
        raise ValueError("Please upload a .csv, .xlsx, or .xls dataset.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _make_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", encoder)])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def _prepare(df, target):
    if target not in df.columns:
        raise ValueError(f"Dataset must contain target column '{target}'.")
    work = df.dropna(subset=[target]).copy()
    y = work[target]
    X = work.drop(columns=[c for c in [TARGET_CLASS, TARGET_REG, *DROP_COLUMNS] if c in work.columns], errors="ignore")
    # Avoid date strings being treated as high-cardinality categories.
    for col in X.columns:
        if "date" in col.lower() and not pd.api.types.is_numeric_dtype(X[col]):
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().mean() > 0.5:
                X[col + "_year"] = parsed.dt.year
                X[col + "_month"] = parsed.dt.month
                X[col + "_day"] = parsed.dt.day
                X = X.drop(columns=[col])
    return X, y


def train_classification(df, test_size=0.2, random_state=42):
    X, y = _prepare(df, TARGET_CLASS)
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError("completion_status must contain both classes (0 and 1).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    prep = _make_preprocessor(X_train)
    estimators = [
        ("logistic", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ("random_forest", RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)),
        ("adaboost", AdaBoostClassifier(n_estimators=80, learning_rate=0.5, random_state=random_state)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100, random_state=random_state)),
    ]
    try:
        from xgboost import XGBClassifier
        estimators.append(("xgboost", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_state, n_jobs=2)))
    except ImportError:
        pass
    models = {name: Pipeline([("prep", clone(prep)), ("model", model)]) for name, model in estimators}
    models["bagging"] = Pipeline([("prep", clone(prep)), ("model", BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8, random_state=random_state), n_estimators=60, random_state=random_state, n_jobs=-1))])
    voting_estimators = [(n, clone(m)) for n, m in models.items() if n in ("logistic", "random_forest", "adaboost")]
    models["voting_soft"] = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)
    models["stacking"] = StackingClassifier(estimators=[(n, clone(m)) for n, m in voting_estimators], final_estimator=LogisticRegression(max_iter=1200), cv=3, n_jobs=-1)
    results, fitted = [], {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, proba) if proba is not None and y_test.nunique() == 2 else np.nan})
            fitted[name] = model
        except Exception as exc:
            results.append({"Model": name, "Error": str(exc)[:180]})
    table = pd.DataFrame(results)
    valid_table = table.dropna(subset=["F1-score"]) if "F1-score" in table else pd.DataFrame()
    best_name = valid_table.sort_values("F1-score", ascending=False).iloc[0]["Model"] if not valid_table.empty else next(iter(fitted))
    return {"models": fitted, "results": table, "best_name": best_name, "best_model": fitted[best_name],
            "X_test": X_test, "y_test": y_test, "features": X.columns.tolist(), "target": TARGET_CLASS}


def train_regression(df, test_size=0.2, random_state=42):
    X, y = _prepare(df, TARGET_REG)
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    prep = _make_preprocessor(X_train)
    estimators = [
        ("ridge", Ridge(alpha=1.0)),
        ("random_forest", RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
        ("adaboost", AdaBoostRegressor(n_estimators=80, learning_rate=0.5, random_state=random_state)),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=100, random_state=random_state)),
    ]
    try:
        from xgboost import XGBRegressor
        estimators.append(("xgboost", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=random_state, n_jobs=2)))
    except ImportError:
        pass
    models = {name: Pipeline([("prep", clone(prep)), ("model", model)]) for name, model in estimators}
    models["bagging"] = Pipeline([("prep", clone(prep)), ("model", BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8, random_state=random_state), n_estimators=60, random_state=random_state, n_jobs=-1))])
    voting_estimators = [(n, clone(m)) for n, m in models.items() if n in ("ridge", "random_forest", "adaboost")]
    models["voting"] = VotingRegressor(estimators=voting_estimators, n_jobs=-1)
    models["stacking"] = StackingRegressor(estimators=[(n, clone(m)) for n, m in voting_estimators], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    results, fitted = [], {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": rmse, "R² Score": r2_score(y_test, pred)})
            fitted[name] = model
        except Exception as exc:
            results.append({"Model": name, "Error": str(exc)[:180]})
    table = pd.DataFrame(results)
    valid_table = table.dropna(subset=["RMSE"]) if "RMSE" in table else pd.DataFrame()
    best_name = valid_table.sort_values("RMSE", ascending=True).iloc[0]["Model"] if not valid_table.empty else next(iter(fitted))
    return {"models": fitted, "results": table, "best_name": best_name, "best_model": fitted[best_name],
            "X_test": X_test, "y_test": y_test, "features": X.columns.tolist(), "target": TARGET_REG}

# Import these after helper definitions to keep the estimator declarations readable.
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
