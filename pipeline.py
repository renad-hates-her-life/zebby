import os
import pandas as pd
import joblib

from kfp.v2.dsl import (
    component,
    Input, Output,
    Dataset, Model, Markdown,
    pipeline
)
from model_utils import (
    preprocess_df,
    auto_type_and_encode,
    split_data,
    train_pipeline  # unchanged
)

# ─────────────────────────────────────────────────────────────────────────────
# 1) get_raw_data: fetch CSV and also pickle the DataFrame
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.9-slim",
    target_image=os.getenv("IMAGE_URI"),
    packages_to_install=["pandas", "boto3", "joblib"]
)
def get_raw_data(
    minio_endpoint:   str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_region:     str,
    bucket:           str,
    object_key:       str,
    table_df:         Output[Dataset],  # CSV
    table_df_pkl:     Output[Dataset]   # Pickle
):
    import pandas as pd
    import boto3
    import os
    import joblib

    # Set env vars so get_object in model_utils (if used) sees them; here we do boto3 inline
    os.environ["MINIO_ENDPOINT"]   = minio_endpoint
    os.environ["MINIO_ACCESS_KEY"] = minio_access_key
    os.environ["MINIO_SECRET_KEY"] = minio_secret_key
    os.environ["MINIO_REGION"]     = minio_region

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name=minio_region
    )
    resp = s3.get_object(Bucket=bucket, Key=object_key)
    df = pd.read_csv(resp["Body"])

    # 1a) Write CSV
    df.to_csv(table_df.path, index=False)
    # 1b) Pickle DataFrame
    joblib.dump(df, table_df_pkl.path)


# ─────────────────────────────────────────────────────────────────────────────
# 2) preprocess: use preprocess_df(), write CSV + pickle cleaned DataFrame
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.9-slim",
    target_image=os.getenv("IMAGE_URI"),
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "category_encoders", "joblib"]
)
def preprocess(
    in_df:     Input[Dataset],
    out_df:    Output[Dataset],  # CSV
    out_df_pkl: Output[Dataset]  # Pickle
):
    import pandas as pd
    import joblib
    from model_utils import preprocess_df

    df = pd.read_csv(in_df.path)
    df_clean = preprocess_df(df)

    # 2a) Write CSV
    df_clean.to_csv(out_df.path, index=False)
    # 2b) Pickle the cleaned DataFrame
    joblib.dump(df_clean, out_df_pkl.path)


# ─────────────────────────────────────────────────────────────────────────────
# 3) feature_engineering: auto‐type/encode + split → write CSVs + pickle each split DataFrame
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.9-slim",
    target_image=os.getenv("IMAGE_URI"),
    packages_to_install=["pandas", "numpy", "scikit-learn", "category_encoders", "lightgbm", "xgboost", "joblib"]
)
def feature_engineering(
    pre:            Input[Dataset],
    train_X:        Output[Dataset],  # CSV
    train_X_pkl:    Output[Dataset],  # Pickle
    train_y:        Output[Dataset],
    train_y_pkl:    Output[Dataset],
    valid_X:        Output[Dataset],
    valid_X_pkl:    Output[Dataset],
    valid_y:        Output[Dataset],
    valid_y_pkl:    Output[Dataset],
    test_X:         Output[Dataset],
    test_X_pkl:     Output[Dataset],
    test_y:         Output[Dataset],
    test_y_pkl:     Output[Dataset],
    validation_size: float = 0.1,
    test_size:       float = 0.1,
    label:           str   = "default"
):
    import pandas as pd
    import joblib
    from model_utils import auto_type_and_encode, split_data

    df = pd.read_csv(pre.path)
    if label not in df.columns:
        raise ValueError(f"Label '{label}' not found in the incoming DataFrame.")

    # 3a) Auto‐type & encode; returns a new DataFrame with label re‐appended
    df_encoded = auto_type_and_encode(df, label)

    # 3b) Split into train/val/test
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df_encoded, label, validation_size, test_size)

    # 3c) Write each split to CSV
    X_tr.to_csv(train_X.path, index=False)
    y_tr.to_csv(train_y.path, index=False)
    X_va.to_csv(valid_X.path, index=False)
    y_va.to_csv(valid_y.path, index=False)
    X_te.to_csv(test_X.path, index=False)
    y_te.to_csv(test_y.path, index=False)

    # 3d) Pickle each split DataFrame/Series
    joblib.dump(X_tr,    train_X_pkl.path)
    joblib.dump(y_tr,    train_y_pkl.path)
    joblib.dump(X_va,    valid_X_pkl.path)
    joblib.dump(y_va,    valid_y_pkl.path)
    joblib.dump(X_te,    test_X_pkl.path)
    joblib.dump(y_te,    test_y_pkl.path)


# ─────────────────────────────────────────────────────────────────────────────
# 4) train_model: train pipeline → extract preprocessor & classifier → write CSV/MD & pickle each piece
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.9-slim",
    target_image=os.getenv("IMAGE_URI"),
    packages_to_install=["pandas", "joblib", "scikit-learn", "xgboost"]
)
def train_model(
    train_X:          Input[Dataset],
    train_X_pkl:      Input[Dataset],
    train_y:          Input[Dataset],
    train_y_pkl:      Input[Dataset],
    valid_X:          Input[Dataset],
    valid_X_pkl:      Input[Dataset],
    valid_y:          Input[Dataset],
    valid_y_pkl:      Input[Dataset],
    preprocessor_out: Output[Dataset],  # Pickle
    classifier_out:   Output[Dataset],  # Pickle
    training_results: Output[Markdown],
    learning_rate:   float = 0.1,
    max_depth:       int   = 6,
    n_estimators:    int   = 100
):
    import pandas as pd
    import joblib
    from model_utils import train_pipeline

    # 4a) Load the split CSVs (we could have used the pickles, but loading CSV is fine)
    X_tr = pd.read_csv(train_X.path)
    y_tr = pd.read_csv(train_y.path).iloc[:, 0]
    X_va = pd.read_csv(valid_X.path)
    y_va = pd.read_csv(valid_y.path).iloc[:, 0]

    # 4b) Fit the pipeline (auto‐scaling, etc.) and get recall
    pipe, recall_val = train_pipeline(X_tr, y_tr, X_va, y_va, learning_rate, max_depth, n_estimators)

    # 4c) Extract fitted preprocessor & trained classifier from `pipe`
    preprocessor = pipe.named_steps["preprocessor"]
    clf =          pipe.named_steps["classifier"]

    # 4d) Pickle each: preprocessor and classifier
    joblib.dump(preprocessor, preprocessor_out.path)
    joblib.dump(clf,          classifier_out.path)

    # 4e) Write out the validation recall Markdown
    with open(training_results.path, "w") as f:
        f.write(f"**Validation Recall:** {recall_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5) test_model: load preprocessor & classifier pickles → evaluate + write CSV/MD and pickle test predictions
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.9-slim",
    target_image=os.getenv("IMAGE_URI"),
    packages_to_install=["pandas", "joblib", "scikit-learn"]
)
def test_model(
    test_X:            Input[Dataset],
    test_X_pkl:        Input[Dataset],
    test_y:            Input[Dataset],
    test_y_pkl:        Input[Dataset],
    preprocessor_in:   Input[Dataset],
    classifier_in:     Input[Dataset],
    test_results:      Output[Markdown],
    test_preds_pkl:    Output[Dataset]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import recall_score

    # 5a) Load test split CSVs (or you could load the pickles)
    X_te = pd.read_csv(test_X.path)
    y_te = pd.read_csv(test_y.path).iloc[:, 0]

    # 5b) Load each artifact
    preprocessor = joblib.load(preprocessor_in.path)
    clf          = joblib.load(classifier_in.path)

    # 5c) Transform test set and predict
    X_te_trans = preprocessor.transform(X_te)
    preds_proba = clf.predict_proba(X_te_trans)[:, 1]
    y_pred      = clf.predict(X_te_trans)
    recall_te   = recall_score(y_te, y_pred)

    # 5d) Write recall to Markdown
    with open(test_results.path, "w") as f:
        f.write(f"**Test Recall:** {recall_te:.4f}")

    # 5e) Pickle the predicted probabilities (or predicted labels) for later analysis
    joblib.dump(preds_proba, test_preds_pkl.path)


# ─────────────────────────────────────────────────────────────────────────────
# 6) Full pipeline definition—wire up all new outputs
# ─────────────────────────────────────────────────────────────────────────────
@pipeline(
    name="credit-scoring-full-pickle-pipeline",
    description="Every step writes both CSV (or MD) + a .pkl of the Python object."
)
def credit_scoring_pipeline(
    minio_endpoint:    str = os.getenv("MINIO_ENDPOINT"),
    minio_access_key:  str = os.getenv("MINIO_ACCESS_KEY"),
    minio_secret_key:  str = os.getenv("MINIO_SECRET_KEY"),
    minio_region:      str = os.getenv("MINIO_REGION"),
    bucket:            str = os.getenv("BUCKET_NAME"),
    object_key:        str = "dummy_raw.csv",
    validation_size:   float = 0.1,
    test_size:         float = 0.1,
    learning_rate:     float = 0.1,
    max_depth:         int   = 6,
    n_estimators:      int   = 100
):
    # 1) get_raw_data
    raw = get_raw_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_region=minio_region,
        bucket=bucket,
        object_key=object_key
    )

    # 2) preprocess
    pre = preprocess(
        in_df=raw.outputs["table_df"],
        out_df=raw.outputs["table_df"] + "_clean.csv",        # arbitrary CSV path
        out_df_pkl=raw.outputs["table_df_pkl"] + "_clean.pkl" # arbitrary PKL path
    )

    # 3) feature_engineering
    feat = feature_engineering(
        pre=pre.outputs["out_df"],
        train_X=    "/mnt/data/train_X.csv",    train_X_pkl=    "/mnt/data/train_X.pkl",
        train_y=    "/mnt/data/train_y.csv",    train_y_pkl=    "/mnt/data/train_y.pkl",
        valid_X=    "/mnt/data/valid_X.csv",    valid_X_pkl=    "/mnt/data/valid_X.pkl",
        valid_y=    "/mnt/data/valid_y.csv",    valid_y_pkl=    "/mnt/data/valid_y.pkl",
        test_X=     "/mnt/data/test_X.csv",     test_X_pkl=     "/mnt/data/test_X.pkl",
        test_y=     "/mnt/data/test_y.csv",     test_y_pkl=     "/mnt/data/test_y.pkl",
        validation_size=validation_size,
        test_size=test_size,
        label="default"
    )

    # 4) train_model
    train = train_model(
        train_X=feat.outputs["train_X"],
        train_X_pkl=feat.outputs["train_X_pkl"],
        train_y=feat.outputs["train_y"],
        train_y_pkl=feat.outputs["train_y_pkl"],
        valid_X=feat.outputs["valid_X"],
        valid_X_pkl=feat.outputs["valid_X_pkl"],
        valid_y=feat.outputs["valid_y"],
        valid_y_pkl=feat.outputs["valid_y_pkl"],
        preprocessor_out="/mnt/data/preprocessor.pkl",
        classifier_out="/mnt/data/classifier.pkl",
        training_results="/mnt/data/train_results.md"
    )

    # 5) test_model
    _ = test_model(
        test_X=feat.outputs["test_X"],
        test_X_pkl=feat.outputs["test_X_pkl"],
        test_y=feat.outputs["test_y"],
        test_y_pkl=feat.outputs["test_y_pkl"],
        preprocessor_in=train.outputs["preprocessor_out"],
        classifier_in=train.outputs["classifier_out"],
        test_results="/mnt/data/test_results.md",
        test_preds_pkl="/mnt/data/test_preds.pkl"
    )

