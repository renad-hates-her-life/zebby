# model_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PowerTransformer,
    KBinsDiscretizer
)
from sklearn.metrics import recall_score, roc_auc_score
from xgboost import XGBClassifier
import category_encoders as ce


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Handle missing values:
       - Numeric columns: if <10% missing → median impute;
                          else → predict missing via RandomForestRegressor (fallback to median).
       - Categorical columns: if <10% missing → fill with mode;
                              else → fill with "__MISSING__".
    2) Handle outliers for each numeric column:
       - Use IQR rule to detect, then cap values at the 1st/99th percentiles.
    Returns a cleaned copy of df.
    """
    df = df.copy()

    # Identify numeric vs. categorical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1a) Numeric missing‐value imputation
    for col in num_cols:
        pct_missing = df[col].isna().mean()
        if pct_missing == 0:
            continue

        if pct_missing < 0.10:
            # Simple median imputation
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            # Attempt to predict missing with RandomForestRegressor
            non_missing = df[df[col].notna()]
            missing_mask = df[col].isna()
            if len(non_missing) >= 10 and missing_mask.sum() > 0:
                feat_cols = [c for c in num_cols + cat_cols if c != col]
                train_data = non_missing.dropna(subset=feat_cols)
                if len(train_data) >= 10:
                    # Encode categorical predictors as integer codes
                    train_X = train_data[feat_cols].copy()
                    for c in cat_cols:
                        train_X[c] = train_X[c].astype("category").cat.codes
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(train_X, train_data[col])

                    miss_data = df[missing_mask]
                    miss_X = miss_data[feat_cols].copy()
                    for c in cat_cols:
                        miss_X[c] = miss_X[c].astype("category").cat.codes
                    df.loc[missing_mask, col] = rf.predict(miss_X)
                else:
                    # Fallback to median if not enough training rows
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
            else:
                # Fallback to median if not enough non‐missing data
                med = df[col].median()
                df[col] = df[col].fillna(med)

    # 1b) Categorical missing‐value imputation
    for col in cat_cols:
        pct_missing = df[col].isna().mean()
        if pct_missing == 0:
            continue

        if pct_missing < 0.10:
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
        else:
            df[col] = df[col].fillna("__MISSING__")

    # 2) Outlier handling for numeric columns
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)

        df[col] = np.where(df[col] < lower_bound, p1, df[col])
        df[col] = np.where(df[col] > upper_bound, p99, df[col])

    return df


def auto_type_and_encode(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    1) For each feature f ≠ label:
         - Generate candidate encodings: raw numeric, quantile discretized, frequency encoded, target encoded.
         - Compute proxy‐Gini = 2·AUC – 1 via 3‐fold stratified CV for each encoding.
         - Apply expert rules (Table 1) to pick feature type/encoding.
    2) If column is datetime-like (>90% parseable), convert to integer timestamp.
    3) Return a new DataFrame where each feature column is replaced by its chosen numeric encoding,
       plus the label column untouched.
    """
    df = df.copy()
    y = df[label].values
    features = [c for c in df.columns if c != label]

    # Unique counts & a helper to detect datetimes
    unique_counts = {c: df[c].nunique(dropna=True) for c in features}

    def is_datetime(col: str) -> bool:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            return parsed.notna().mean() > 0.90
        except:
            return False

    encoded_cols = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for col in features:
        X_col = df[[col]].copy()
        candidates = {}

        # 1) Raw numeric (if dtype is numeric)
        if pd.api.types.is_numeric_dtype(X_col[col]):
            candidates["raw"] = X_col[col].values.reshape(-1, 1)
        else:
            candidates["raw"] = None

        # 2) Quantile discretization
        try:
            kb = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
            disc = kb.fit_transform(X_col)
            candidates["qdisc"] = disc.reshape(-1, 1)
        except:
            candidates["qdisc"] = None

        # 3) Frequency encoding
        freq_map = df[col].value_counts(normalize=True).to_dict()
        freq_encoded = np.array([freq_map.get(v, 0) for v in df[col]]).reshape(-1, 1)
        candidates["freq"] = freq_encoded

        # 4) OOF target encoding
        te_series = pd.Series(index=df.index, dtype=float)
        for train_i, val_i in skf.split(df, y):
            tr, va = df.iloc[train_i], df.iloc[val_i]
            means = tr.groupby(col)[label].mean()
            te_series.iloc[val_i] = va[col].map(means)
        te_series.fillna(df[label].mean(), inplace=True)
        candidates["oof"] = te_series.values.reshape(-1, 1)

        # Compute proxy‐Gini for each candidate
        scores = {}
        for name, arr in candidates.items():
            if arr is None:
                scores[name] = -np.inf
                continue
            aucs = []
            for train_i, val_i in skf.split(arr, y):
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(solver="liblinear")
                lr.fit(arr[train_i], y[train_i])
                prob = lr.predict_proba(arr[val_i])[:, 1]
                aucs.append(roc_auc_score(y[val_i], prob))
            scores[name] = 2 * np.mean(aucs) - 1

        Graw = scores["raw"]
        Gqdisc = scores["qdisc"]
        Gfreq = scores["freq"]
        Goof  = scores["oof"]
        frac_unique = unique_counts[col] / len(df)

        # Expert rules (Table 1)
        chosen = "raw"
        if Graw > max(Gqdisc, Gfreq, Goof) and\
           frac_unique > 0.05 and pd.api.types.is_numeric_dtype(df[col]):
            chosen = "raw"
        elif max(Gfreq, Goof) > Graw and\
             unique_counts[col] < 50 and not pd.api.types.is_numeric_dtype(df[col]):
            chosen = "oof" if Goof >= Gfreq else "freq"
        elif Gqdisc > max(Graw, Gfreq, Goof) and unique_counts[col] < 10:
            chosen = "qdisc"
        elif Gfreq > Graw and frac_unique > 0.10 and not pd.api.types.is_numeric_dtype(df[col]):
            chosen = "freq"
        elif is_datetime(col):
            # Convert to integer timestamp and skip encoding
            df[col] = pd.to_datetime(df[col], errors="coerce").astype(int)
            encoded_cols[col] = df[col].values
            continue
        else:
            chosen = max(scores, key=lambda k: scores[k])

        encoded_cols[col] = candidates[chosen].ravel()

    # Build final DataFrame
    df_out = pd.DataFrame({c: encoded_cols[c] for c in encoded_cols}, index=df.index)
    df_out[label] = df[label].values
    return df_out


def split_data(
    df: pd.DataFrame,
    label: str,
    validation_size: float,
    test_size: float,
    random_state: int = 42
):
    """
    Splits df into train / validation / test. Returns:
      X_tr, X_va, X_te, y_tr, y_va, y_te
    """
    X = df.drop(columns=[label])
    y = df[label]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=validation_size + test_size, random_state=random_state
    )
    rel_test = test_size / (validation_size + test_size)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=random_state
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def train_pipeline(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    learning_rate: float,
    max_depth: int,
    n_estimators: int
):
    """
    Builds a ColumnTransformer that chooses per‐column scaling:
      - If a numeric column’s skewness >1 → PowerTransformer → StandardScaler
      - Else → SimpleImputer(median) → StandardScaler
    Then an XGBClassifier is trained. Returns (fitted_pipeline, validation_recall).
    """
    numeric_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()

    transformers = []
    for col in numeric_cols:
        skew = X_tr[col].skew()
        if abs(skew) > 1.0:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("power",   PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale",   StandardScaler())
            ])
        else:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale",   StandardScaler())
            ])
        transformers.append((f"num_{col}", num_pipe, [col]))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf)
    ])

    pipe.fit(X_tr, y_tr)
    y_pred_va = pipe.predict(X_va)
    recall_val = recall_score(y_va, y_pred_va)

    return pipe, recall_val

