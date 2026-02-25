"""
═══════════════════════════════════════════
 ML Engine v2 — Maximum Accuracy Pipeline
 12+ Algorithms · Semantic Analysis · Fuzzy Matching · Auto-Tuning
═══════════════════════════════════════════
"""

import os
import math
import uuid
import joblib
import warnings
import difflib
import re
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

warnings.filterwarnings("ignore")


# ══════════════════════════════════
#  SEMANTIC COLUMN ANALYSER
#  Understands what columns MEAN
# ══════════════════════════════════

# Domain keyword maps — helps understand column meaning
SEMANTIC_DOMAINS = {
    "financial":   ["price", "cost", "revenue", "profit", "loss", "salary", "wage", "income",
                    "amount", "total", "fee", "charge", "tax", "discount", "budget", "spend",
                    "earning", "payment", "rupee", "inr", "usd", "dollar", "₹", "$"],
    "temporal":    ["date", "time", "year", "month", "day", "hour", "minute", "week",
                    "period", "quarter", "timestamp", "created", "updated", "dob", "age"],
    "geographic":  ["city", "state", "country", "region", "zone", "district", "pincode",
                    "pin", "area", "location", "address", "latitude", "longitude", "lat", "lng"],
    "identity":    ["id", "code", "number", "no", "ref", "key", "index", "serial", "sku"],
    "categorical": ["type", "category", "class", "group", "segment", "tier", "level",
                    "status", "gender", "grade", "rank", "tag", "label"],
    "target":      ["target", "label", "output", "result", "outcome", "predict", "churn",
                    "default", "fraud", "risk", "score", "rating", "approved", "converted"],
    "text_free":   ["name", "description", "comment", "note", "remark", "feedback",
                    "review", "title", "summary", "message"],
    "boolean":     ["is_", "has_", "flag", "active", "enabled", "verified", "approved",
                    "churned", "defaulted"],
}

def get_semantic_domain(col_name: str) -> str:
    """Detect the semantic domain of a column from its name."""
    col_lower = col_name.lower().replace("_", " ").replace("-", " ")
    for domain, keywords in SEMANTIC_DOMAINS.items():
        if any(kw in col_lower for kw in keywords):
            return domain
    return "unknown"

def analyze_column_relationships(df: pd.DataFrame, col_types: Dict[str, str]) -> List[Dict]:
    """Find meaningful relationships between columns using correlation + mutual info."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    relationships = []
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]

    if len(numeric_cols) < 2:
        return relationships

    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Pearson correlation
    try:
        corr = num_df.corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr.iloc[i, j]
                if not math.isnan(r) and abs(r) > 0.4:
                    dom_a = get_semantic_domain(numeric_cols[i])
                    dom_b = get_semantic_domain(numeric_cols[j])
                    rel_type = "strong" if abs(r) > 0.7 else "moderate"
                    direction = "positive" if r > 0 else "negative"
                    relationships.append({
                        "col_a": numeric_cols[i],
                        "col_b": numeric_cols[j],
                        "correlation": round(r, 3),
                        "type": rel_type,
                        "direction": direction,
                        "domain_a": dom_a,
                        "domain_b": dom_b,
                        "insight": _relationship_insight(numeric_cols[i], numeric_cols[j], r, dom_a, dom_b)
                    })
    except Exception:
        pass

    return sorted(relationships, key=lambda x: abs(x["correlation"]), reverse=True)[:8]

def _relationship_insight(col_a: str, col_b: str, r: float, dom_a: str, dom_b: str) -> str:
    """Generate plain-English insight about a relationship."""
    direction = "increases" if r > 0 else "decreases"
    strength = "strongly" if abs(r) > 0.7 else "moderately"
    if dom_a == "financial" and dom_b == "financial":
        return f"When {col_a} rises, {col_b} {strength} {direction} — both are financial metrics."
    if dom_a == "temporal" or dom_b == "temporal":
        return f"{col_a} and {col_b} are {strength} correlated over time."
    return f"{col_a} and {col_b} are {strength} {direction[:-1]}d together (r={round(r, 2)})."


# ══════════════════════════════════
#  FUZZY MATCHING CLEANER
#  Removes near-duplicate text values
# ══════════════════════════════════

def clean_fuzzy_categories(df: pd.DataFrame, col_types: Dict[str, str],
                            threshold: float = 0.82) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    For categorical columns: find near-duplicate values (typos, spacing, case issues)
    and standardise them to the most frequent form.
    e.g. ['Mumbai', 'mumbai', 'MUMBAI', 'Mumbay'] → all become 'Mumbai'
    """
    df = df.copy()
    fuzzy_report = []

    cat_cols = [c for c, t in col_types.items() if t in ("categorical", "text") and c in df.columns]

    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) < 2 or len(unique_vals) > 200:
            continue

        # Build mapping: variant → canonical form
        canonical_map = {}
        str_vals = [str(v).strip() for v in unique_vals]
        value_counts = df[col].value_counts()
        merged = []

        for i, val in enumerate(str_vals):
            if val in canonical_map:
                continue
            group = [val]
            for j, other in enumerate(str_vals):
                if i == j or other in canonical_map:
                    continue
                # Compare normalised versions
                norm_val = val.lower().replace(" ", "").replace("-", "").replace("_", "")
                norm_other = other.lower().replace(" ", "").replace("-", "").replace("_", "")
                ratio = difflib.SequenceMatcher(None, norm_val, norm_other).ratio()
                if ratio >= threshold:
                    group.append(other)

            if len(group) > 1:
                # Pick canonical = most frequent in original data
                canonical = max(group, key=lambda v: value_counts.get(v, 0))
                for variant in group:
                    canonical_map[variant] = canonical
                merged.append({"variants": group, "canonical": canonical, "column": col})

        if canonical_map:
            df[col] = df[col].apply(
                lambda x: canonical_map.get(str(x).strip(), x) if pd.notna(x) else x
            )
            fuzzy_report.append({
                "column": col,
                "merges": merged,
                "detail": f"Standardised {len(canonical_map)} fuzzy variants in '{col}'"
            })

    return df, fuzzy_report


# ══════════════════════════════════
#  1. DATA PROFILING
# ══════════════════════════════════

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect column types: numeric, categorical, date, text, boolean, empty."""
    col_types = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            col_types[col] = "empty"
            continue

        # Boolean
        if set(series.unique()).issubset({0, 1, True, False, "true", "false",
                                          "True", "False", "yes", "no", "Yes", "No"}):
            col_types[col] = "boolean"
            continue

        if pd.api.types.is_bool_dtype(series):
            col_types[col] = "boolean"
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05 and series.nunique() <= 20:
                col_types[col] = "categorical"
            else:
                col_types[col] = "numeric"
            continue

        sample = series.head(100).astype(str)

        # Date detection
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(sample) * 0.6:
                col_types[col] = "date"
                continue
        except Exception:
            pass

        # Currency/numeric text
        stripped = sample.str.replace(r"[₹$€£,%\s]", "", regex=True).str.replace(",", "")
        numeric_count = pd.to_numeric(stripped, errors="coerce").notna().sum()
        if numeric_count > len(sample) * 0.7:
            col_types[col] = "numeric"
            continue

        # Semantic domain hint
        domain = get_semantic_domain(col)
        if domain == "identity":
            col_types[col] = "identity"
            continue

        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.3 and series.nunique() <= 60:
            col_types[col] = "categorical"
        else:
            col_types[col] = "text"

    return col_types


def profile_data(df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Any]:
    """Generate comprehensive data profile with semantic insights."""
    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_types": col_types,
        "semantic_domains": {col: get_semantic_domain(col) for col in df.columns},
        "missing": {},
        "missing_total": 0,
        "duplicates": int(df.duplicated().sum()),
        "outliers": {},
        "outliers_total": 0,
        "stats": {},
        "correlations": [],
        "relationships": [],
        "quality_score": 100,
    }

    for col in df.columns:
        missing = int(df[col].isnull().sum() + (df[col].astype(str).str.strip() == "").sum())
        if missing > 0:
            profile["missing"][col] = missing
            profile["missing_total"] += missing

    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    for col in numeric_cols:
        series = pd.to_numeric(
            df[col].astype(str).str.replace(r"[₹$€£,%\s]", "", regex=True).str.replace(",", ""),
            errors="coerce"
        ).dropna()
        if len(series) < 4:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lo) | (series > hi)).sum())

        profile["stats"][col] = {
            "mean":   round(float(series.mean()), 2),
            "std":    round(float(series.std()), 2),
            "min":    round(float(series.min()), 2),
            "max":    round(float(series.max()), 2),
            "q1":     round(q1, 2),
            "median": round(float(series.median()), 2),
            "q3":     round(q3, 2),
            "skew":   round(float(series.skew()), 2),
        }
        if outlier_count > 0:
            profile["outliers"][col] = outlier_count
            profile["outliers_total"] += outlier_count

    # Relationships
    profile["relationships"] = analyze_column_relationships(df, col_types)

    # Correlations for frontend chart
    for r in profile["relationships"]:
        profile["correlations"].append({
            "col_a": r["col_a"], "col_b": r["col_b"], "r": r["correlation"]
        })

    # Quality score
    issue_count = (len(profile["missing"]) + len(profile["outliers"]) +
                   (1 if profile["duplicates"] > 0 else 0))
    profile["quality_score"] = max(0, 100 - issue_count * 8 - min(profile["duplicates"], 5) * 4)

    return profile


# ══════════════════════════════════
#  2. DATA CLEANING (Enhanced)
# ══════════════════════════════════

def clean_data(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced cleaning:
    - Remove duplicates
    - Fuzzy-match and standardise near-duplicate categories
    - Strip currency / special symbols
    - Fill missing values
    - Remove outliers (IQR)
    - Drop identity columns (IDs add no signal)
    """
    report = {"steps": [], "rows_before": len(df)}
    df = df.copy()

    # 1. Drop identity/ID columns — they hurt ML
    id_cols = [c for c, t in col_types.items() if t == "identity" and c in df.columns]
    for col in id_cols:
        df.drop(columns=[col], inplace=True)
        col_types.pop(col, None)
        report["steps"].append({"action": "drop_id", "detail": f"Dropped ID column '{col}' — IDs don't help prediction"})

    # 2. Remove duplicates
    dups = int(df.duplicated().sum())
    if dups > 0:
        df = df.drop_duplicates()
        report["steps"].append({"action": "remove_duplicates", "detail": f"Removed {dups} duplicate rows"})

    # 3. Fuzzy matching — fix typos in categories
    df, fuzzy_report = clean_fuzzy_categories(df, col_types)
    for fr in fuzzy_report:
        report["steps"].append({"action": "fuzzy_clean", "detail": fr["detail"]})

    # 4. Strip currency symbols and convert to numeric
    for col in list(df.columns):
        if col_types.get(col) in ("numeric", "text") and df[col].dtype == object:
            stripped = df[col].astype(str).str.replace(r"[₹$€£,%\s]", "", regex=True).str.replace(",", "")
            numeric_vals = pd.to_numeric(stripped, errors="coerce")
            if numeric_vals.notna().sum() > len(df) * 0.6:
                df[col] = numeric_vals
                col_types[col] = "numeric"
                report["steps"].append({"action": "strip_symbols", "detail": f"Stripped currency/percent symbols from '{col}'"})

    # 5. Fill missing values
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue
        null_pct = missing / len(df)

        if null_pct > 0.6:
            df.drop(columns=[col], inplace=True)
            col_types.pop(col, None)
            report["steps"].append({"action": "drop_column", "detail": f"Dropped '{col}' — {round(null_pct*100)}% missing"})
            continue

        if col_types.get(col) == "numeric":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            report["steps"].append({"action": "fill_missing", "detail": f"Filled {missing} missing values in '{col}' with median ({round(float(fill_val), 2)})"})
        elif col_types.get(col) == "boolean":
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else False)
        else:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report["steps"].append({"action": "fill_missing", "detail": f"Filled {missing} missing values in '{col}' with most common value ('{fill_val}')"})

    # 6. Remove outliers (IQR — only if <15% of data affected)
    numeric_cols = [c for c, t in col_types.items() if t == "numeric" and c in df.columns]
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (series < lo) | (series > hi)
        outlier_count = int(outlier_mask.sum())
        if 0 < outlier_count < len(df) * 0.15:
            df = df[~outlier_mask]
            report["steps"].append({"action": "remove_outliers", "detail": f"Removed {outlier_count} outliers from '{col}' (IQR method)"})

    report["rows_after"] = len(df)
    if not report["steps"]:
        report["steps"].append({"action": "no_changes", "detail": "Data was already clean — no changes needed"})

    return df, report


# ══════════════════════════════════
#  3. FEATURE ENGINEERING (Enhanced)
# ══════════════════════════════════

def engineer_features(df: pd.DataFrame, col_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Advanced feature engineering:
    - Date → rich temporal features (month, day, weekday, quarter, is_weekend)
    - Categorical → smart encoding (OHE for low cardinality, label for high)
    - Boolean → int conversion
    - Numeric → StandardScaler / RobustScaler based on skewness
    - Drop high-cardinality text (no signal)
    - Mutual information feature selection
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
    from sklearn.feature_selection import VarianceThreshold

    report = {"steps": [], "features_before": len(df.columns), "encoders": {}}
    df = df.copy()
    encoders = {}

    # 1. Boolean → int
    bool_cols = [c for c, t in col_types.items() if t == "boolean" and c in df.columns]
    for col in bool_cols:
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        report["steps"].append({"action": "bool_encode", "detail": f"Converted boolean '{col}' to 0/1"})

    # 2. Date features (rich extraction)
    date_cols = [c for c, t in col_types.items() if t == "date" and c in df.columns]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"]      = df[col].dt.year
            df[f"{col}_month"]     = df[col].dt.month
            df[f"{col}_day"]       = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"]   = df[col].dt.quarter
            df[f"{col}_is_weekend"]= (df[col].dt.dayofweek >= 5).astype(int)
            df.drop(columns=[col], inplace=True)
            report["steps"].append({"action": "date_extract", "detail": f"Extracted 6 features from '{col}' (year, month, day, weekday, quarter, is_weekend)"})
        except Exception:
            pass

    # 3. Categorical encoding
    cat_cols = [c for c, t in col_types.items() if t == "categorical" and c in df.columns]
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 1:
            df.drop(columns=[col], inplace=True)
            continue
        if n_unique <= 10:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            report["steps"].append({"action": "ohe", "detail": f"One-hot encoded '{col}' ({n_unique} categories → {len(dummies.columns)} columns)"})
        else:
            # Label encode
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df.drop(columns=[col], inplace=True)
            report["steps"].append({"action": "label_encode", "detail": f"Label encoded '{col}' ({n_unique} unique values)"})

    # 4. Drop high-cardinality text columns (no signal for ML)
    text_cols = [c for c, t in col_types.items() if t == "text" and c in df.columns]
    target_hint = [c for c in df.columns if get_semantic_domain(c) == "target"]
    for col in text_cols:
        if col in target_hint:
            continue
        if df[col].nunique() > 50:
            df.drop(columns=[col], inplace=True)
            report["steps"].append({"action": "drop_text", "detail": f"Dropped high-cardinality text '{col}' — too many unique values for ML"})
        else:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df.drop(columns=[col], inplace=True)
            report["steps"].append({"action": "label_encode", "detail": f"Label encoded text '{col}'"})

    # 5. Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = df.columns[-1]
    scale_cols = [c for c in numeric_cols if c != target_col and "_encoded" not in c]

    if scale_cols:
        # Choose scaler based on skewness
        skewnesses = [abs(float(df[c].skew())) for c in scale_cols if df[c].std() > 0]
        avg_skew = np.mean(skewnesses) if skewnesses else 0

        if avg_skew > 1.5:
            scaler = RobustScaler()
            scaler_name = "RobustScaler (handles skewed data)"
        else:
            scaler = StandardScaler()
            scaler_name = "StandardScaler (normal distribution)"

        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        encoders["__scaler__"] = scaler
        report["steps"].append({"action": "scale", "detail": f"{scaler_name} applied to {len(scale_cols)} numeric columns"})

    # 6. Remove zero-variance features
    try:
        vt = VarianceThreshold(threshold=0.0)
        feat_cols = [c for c in df.columns if c != target_col]
        num_feat_df = df[feat_cols].select_dtypes(include=[np.number])
        if len(num_feat_df.columns) > 1:
            vt.fit(num_feat_df.fillna(0))
            zero_var = [c for c, keep in zip(num_feat_df.columns, vt.get_support()) if not keep]
            if zero_var:
                df.drop(columns=zero_var, inplace=True)
                report["steps"].append({"action": "drop_zero_var", "detail": f"Dropped {len(zero_var)} zero-variance features"})
    except Exception:
        pass

    df.dropna(inplace=True)
    report["features_after"] = len(df.columns)
    return df, encoders, report


# ══════════════════════════════════
#  4. PROBLEM TYPE DETECTION
# ══════════════════════════════════

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    target = df[target_col]
    if target.dtype == object or target.dtype.name == "category":
        return "classification"
    if set(target.dropna().unique()).issubset({0, 1}):
        return "classification"
    unique_ratio = target.nunique() / len(target)
    if target.nunique() <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


# ══════════════════════════════════
#  5. MODEL TRAINING — 12+ ALGORITHMS
# ══════════════════════════════════

def _get_classifiers():
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs"),
        "KNN":                 KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Neural Network":      MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        )
    except ImportError:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, depth=6, random_seed=42,
            verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass

    return models


def _get_regressors():
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        ExtraTreesRegressor, AdaBoostRegressor,
    )
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor

    models = {
        "Random Forest":    RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        "Extra Trees":      ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42),
        "AdaBoost":         AdaBoostRegressor(n_estimators=100, random_state=42),
        "Decision Tree":    DecisionTreeRegressor(max_depth=10, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=2000),
        "ElasticNet":       ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        "KNN":              KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        "Neural Network":   MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, random_state=42,
            verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=200, depth=6, random_seed=42,
            verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass

    return models


def train_models(df: pd.DataFrame, target_col: str, problem_type: str) -> Dict[str, Any]:
    """
    Train ALL available algorithms, cross-validate each,
    fine-tune the top-3, return the best.
    """
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, f1_score,
        r2_score, mean_squared_error, mean_absolute_error,
    )

    # Prepare data
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col].copy()

    if len(X.columns) == 0:
        raise ValueError("No numeric features available after engineering.")
    if len(X) < 10:
        raise ValueError("Not enough rows to train (need at least 10).")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Encode target for classification
    le_target = None
    if problem_type == "classification":
        if y.dtype == object or y.dtype.name == "category":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None
    )

    scoring = "accuracy" if problem_type == "classification" else "r2"
    n_cv = min(5, max(2, len(X_train) // 20))

    models = _get_classifiers() if problem_type == "classification" else _get_regressors()

    # ── Phase 1: Quick CV on all models ──
    results = []
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_cv,
                                         scoring=scoring, n_jobs=-1)
            mean_s = float(np.mean(cv_scores))
            std_s  = float(np.std(cv_scores))
            results.append({"name": name, "model": model,
                             "score": round(mean_s, 4), "std": round(std_s, 4)})
        except Exception as e:
            results.append({"name": name, "model": None,
                             "score": -1, "std": 0, "error": str(e)})

    results_valid = [r for r in results if r["score"] >= 0]
    results_sorted = sorted(results_valid, key=lambda x: -x["score"])

    if not results_sorted:
        raise ValueError("All models failed. Please check your data.")

    # ── Phase 2: Fine-tune top 3 ──
    TUNE_PARAMS = {
        "Random Forest":     {"n_estimators": [100, 200, 300], "max_depth": [None, 8, 15], "min_samples_split": [2, 5]},
        "Extra Trees":       {"n_estimators": [100, 200, 300], "max_depth": [None, 8, 15]},
        "XGBoost":           {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2]},
        "LightGBM":          {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        "Gradient Boosting": {"n_estimators": [100, 150], "max_depth": [3, 5, 7]},
    }

    for r in results_sorted[:3]:
        name = r["name"]
        if name in TUNE_PARAMS and r["model"] is not None:
            try:
                search = RandomizedSearchCV(
                    r["model"], TUNE_PARAMS[name],
                    n_iter=8, cv=n_cv, scoring=scoring,
                    random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                tuned_score = float(search.best_score_)
                if tuned_score > r["score"]:
                    r["score"] = round(tuned_score, 4)
                    r["model"] = search.best_estimator_
                    r["tuned"] = True
            except Exception:
                pass

    # Re-sort after tuning
    results_sorted = sorted(results_valid, key=lambda x: -x["score"])
    best = results_sorted[0]
    best_model = best["model"]

    # ── Train best on full training set ──
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    # Evaluation
    if problem_type == "classification":
        eval_metrics = {
            "accuracy":    round(float(accuracy_score(y_test, predictions)), 4),
            "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        }
    else:
        eval_metrics = {
            "r2":   round(float(r2_score(y_test, predictions)), 4),
            "mae":  round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        }

    # Feature importance
    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        for fname, imp in sorted(
            zip(X.columns, best_model.feature_importances_),
            key=lambda x: -x[1]
        )[:10]:
            feature_importance.append({"feature": fname, "importance": round(float(imp), 4)})
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_).flatten() if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
        for fname, imp in sorted(zip(X.columns, coefs), key=lambda x: -x[1])[:10]:
            feature_importance.append({"feature": fname, "importance": round(float(imp), 4)})

    # Mutual information insights
    mi_insights = _mutual_information_insights(X_train, y_train, problem_type, list(X.columns))

    explanation = _generate_explanation(
        best["name"], problem_type, best["score"],
        feature_importance, len(X.columns), best.get("tuned", False),
        eval_metrics
    )

    # All scores (without model objects for serialization)
    all_scores = [
        {"name": r["name"], "score": r["score"], "std": r["std"],
         "tuned": r.get("tuned", False)}
        for r in results_sorted
    ]

    return {
        "all_scores":      all_scores,
        "best_model_name": best["name"],
        "best_cv_score":   best["score"],
        "eval_metrics":    eval_metrics,
        "feature_importance": feature_importance,
        "mi_insights":     mi_insights,
        "explanation":     explanation,
        "problem_type":    problem_type,
        "target_column":   target_col,
        "train_size":      len(X_train),
        "test_size":       len(X_test),
        "model":           best_model,
        "feature_columns": list(X.columns),
        "algorithms_tested": len(results_valid),
    }


def _mutual_information_insights(X, y, problem_type: str, col_names: List[str]) -> List[Dict]:
    """Rank features by mutual information with target."""
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        fn = mutual_info_classif if problem_type == "classification" else mutual_info_regression
        mi = fn(X, y, random_state=42)
        ranked = sorted(zip(col_names, mi), key=lambda x: -x[1])[:5]
        return [
            {"feature": f, "mi_score": round(float(s), 4),
             "insight": f"'{f}' carries {'high' if s > 0.1 else 'moderate'} information about the target"}
            for f, s in ranked if s > 0
        ]
    except Exception:
        return []


def _generate_explanation(name: str, problem_type: str, score: float,
                           fi: List[Dict], n_features: int,
                           tuned: bool, metrics: Dict) -> str:
    tuned_str = " (fine-tuned with hyperparameter search)" if tuned else ""
    top_3 = ", ".join([f["feature"] for f in fi[:3]]) if fi else "your features"
    metric_str = (f"{round(score*100, 1)}% accuracy" if problem_type == "classification"
                  else f"R² = {round(score*100, 1)}%")

    reasons = {
        "Random Forest":     "handles non-linear patterns, resistant to overfitting, great for mixed data types",
        "Extra Trees":       "faster than Random Forest with similar accuracy, reduces variance by using random thresholds",
        "XGBoost":           "industry's top performer on tabular data — boosting + regularisation prevents overfitting",
        "LightGBM":          "fastest gradient boosting, excellent on large datasets with many features",
        "CatBoost":          "handles categorical variables natively, very strong on structured business data",
        "Gradient Boosting": "builds models sequentially, each correcting the last — excellent for complex patterns",
        "Neural Network":    "learns deep non-linear patterns, strong when relationships between features are complex",
        "Logistic Regression": "fast and interpretable, works best when relationships are roughly linear",
        "Ridge Regression":  "linear model with regularisation, best when features are correlated",
        "SVM":               "finds the optimal boundary between classes, effective on smaller datasets",
        "KNN":               "predicts based on similar past examples, effective when local patterns matter",
        "AdaBoost":          "focuses on hard-to-classify examples, good for imbalanced data",
    }
    reason = reasons.get(name, "achieved the highest cross-validated score on your data")

    return (
        f"We tested {n_features} features across all available algorithms. "
        f"{name}{tuned_str} won because it {reason}. "
        f"It achieved {metric_str} using cross-validation. "
        f"The most predictive features in your data are: {top_3}. "
        f"This means your data has clear patterns the model can reliably learn from."
    )


# ══════════════════════════════════
#  6. PYTHON CODE GENERATOR
# ══════════════════════════════════

def generate_full_pipeline_code(
    col_types: Dict[str, str],
    profile: Dict,
    clean_report: Dict,
    eng_report: Dict,
    train_result: Dict,
    filename: str = "your_data.csv",
) -> str:
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    target    = train_result["target_column"]
    prob_type = train_result["problem_type"]
    best_model = train_result["best_model_name"]
    is_cls = prob_type == "classification"

    code = f'''#!/usr/bin/env python3
"""
 Auto-Generated Pipeline by Vibe ML
 Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}
 File      : {filename}
 Rows      : {profile["rows"]} → {clean_report.get("rows_after", "?")} (after cleaning)
 Target    : {target} ({prob_type})
 Best Model: {best_model} ({train_result["best_cv_score"]:.1%} CV score)
 Algorithms tested: {train_result.get("algorithms_tested", "?")}

 To run:
   pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost joblib
   python pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import {"accuracy_score, f1_score, classification_report" if is_cls else "r2_score, mean_squared_error, mean_absolute_error"}
import joblib

try:
    from xgboost import {"XGBClassifier" if is_cls else "XGBRegressor"}
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Load Data ──
df = pd.read_csv("{filename}")
print(f"Loaded: {{df.shape[0]}} rows × {{df.shape[1]}} columns")

# ── Clean ──
df = df.drop_duplicates()
numeric_cols = {repr(num_cols)}
for col in numeric_cols:
    if col in df.columns and df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].str.replace(r"[₹$€£,%\\s]", "", regex=True), errors="coerce")
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]

# ── Feature Engineering ──
target_col = "{target}"
'''

    if date_cols:
        for col in date_cols:
            code += f'''
df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")
df["{col}_year"] = df["{col}"].dt.year
df["{col}_month"] = df["{col}"].dt.month
df["{col}_dayofweek"] = df["{col}"].dt.dayofweek
df["{col}_quarter"] = df["{col}"].dt.quarter
df["{col}_is_weekend"] = (df["{col}"].dt.dayofweek >= 5).astype(int)
df.drop("{col}", axis=1, inplace=True)
'''

    if cat_cols:
        code += "\nle = LabelEncoder()\n"
        for col in cat_cols:
            code += f'if "{col}" in df.columns: df["{col}"] = le.fit_transform(df["{col}"].astype(str))\n'

    code += f'''
df.dropna(inplace=True)
X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
y = df[target_col]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Train: {{len(X_train)}} | Test: {{len(X_test)}}")

# ── Train All Models ──
models = {{
    "Random Forest": {"RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)" if is_cls else "RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)"},
    "Extra Trees":   {"ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)" if is_cls else "ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)"},
    "Gradient Boosting": {"GradientBoostingClassifier(n_estimators=150, random_state=42)" if is_cls else "GradientBoostingRegressor(n_estimators=150, random_state=42)"},
}}
if HAS_XGB:
    models["XGBoost"] = {"XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0)" if is_cls else "XGBRegressor(n_estimators=200, random_state=42, verbosity=0)"}
if HAS_LGB:
    models["LightGBM"] = {"lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)" if is_cls else "lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)"}

best_score, best_name, best_model = {"0" if is_cls else "-999"}, "", None
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="{"accuracy" if is_cls else "r2"}")
    print(f"  {{name}}: {{scores.mean():.4f}} (+/- {{scores.std():.4f}})")
    if scores.mean() > best_score:
        best_score, best_name, best_model = scores.mean(), name, model

print(f"\\n🏆 Best: {{best_name}} ({{best_score:.4f}})")
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)

# ── Evaluate ──
{"print(f'Accuracy: {accuracy_score(y_test, preds):.4f}')" if is_cls else "print(f'R²: {r2_score(y_test, preds):.4f}')"}
{"print(classification_report(y_test, preds, zero_division=0))" if is_cls else "print(f'MAE: {mean_absolute_error(y_test, preds):.4f}')"}

# ── Save ──
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("clean_data.csv", index=False)
print("\\n✅ Saved: model.pkl, scaler.pkl, clean_data.csv")
'''
    return code


# ══════════════════════════════════
#  7. SAVE ARTIFACTS
# ══════════════════════════════════

def save_pipeline_artifacts(
    session_id: str,
    model,
    clean_df: pd.DataFrame,
    code: str,
    encoders: dict,
) -> Dict[str, str]:
    base_dir = os.path.join("storage", "outputs", session_id)
    os.makedirs(base_dir, exist_ok=True)

    model_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(model, model_path)

    enc_path = os.path.join(base_dir, "encoders.pkl")
    joblib.dump(encoders, enc_path)

    data_path = os.path.join(base_dir, "clean_data.csv")
    clean_df.to_csv(data_path, index=False)

    code_path = os.path.join(base_dir, "vibeml_pipeline.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    return {"model": model_path, "encoders": enc_path,
            "data": data_path, "code": code_path}
