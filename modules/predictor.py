import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

FEATURE_COLS = [
    "positive_score",
    "negative_score",
    "neutral_score",
    "polarity",
    "sentiment_confidence",
    "optimism_ratio",
    "caution_ratio",
    "transcript_length",
    "word_complexity",
    "price_volatility",
    "prev_quarter_movement"
]


def load_features():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "features", "feature_table.csv"))
    return df


def prepare_data(df):
    X = df[FEATURE_COLS].copy()
    y = df["label"].copy()

    # split into 80% training and 20% testing
    # random_state=42 ensures the same split every time you run this
    # stratify=y ensures both splits have the same ratio of 0s and 1s
    # without stratify you might get a test set with mostly 1s which
    # would give misleading accuracy scores
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # scale_pos_weight handles our class imbalance
    # we have 427 positive and 315 negative labels
    # scale_pos_weight = negative count / positive count
    # this tells xgboost to pay more attention to the minority class
    scale_pos_weight = 315 / 427

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20
    )

    # we pass X_test and y_test as eval_set so xgboost can monitor
    # performance on unseen data during training
    # early_stopping_rounds=20 means if performance does not improve
    # for 20 consecutive rounds xgboost stops training automatically
    # this prevents overfitting — the model memorising training data
    # instead of learning general patterns
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )

    model.fit(
        X_train_main,
        y_train_main,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"accuracy: {accuracy:.4f}")
    print(f"f1 score: {f1:.4f}")
    print(f"auc-roc: {auc:.4f}")
    print(f"confusion matrix:")
    print(f"  true negative: {cm[0][0]}  false positive: {cm[0][1]}")
    print(f"  false negative: {cm[1][0]}  true positive: {cm[1][1]}")
    print(f"classification report:")
    print(classification_report(y_test, y_pred, target_names=["down", "up"]))

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4)
    }


def get_feature_importance(model):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print("feature importance:")
    for _, row in feature_importance_df.iterrows():
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")

    return feature_importance_df


def save_model(model):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
    joblib.dump(model, model_path)
    print(f"model saved to {model_path}")


def load_saved_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
    model = joblib.load(model_path)
    return model


def predict_single(features_dict):
    # this function is called at runtime when a user searches a stock
    # features_dict is a dictionary of feature values for one transcript
    # returns prediction (0 or 1) and confidence percentage
    model = load_saved_model()

    input_df = pd.DataFrame([features_dict])[FEATURE_COLS]
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0]

    return {
        "prediction": int(prediction),
        "direction": "up" if prediction == 1 else "down",
        "confidence": round(float(max(confidence)) * 100, 2)
    }


def run():
    df = load_features()
    print(f"loaded {len(df)} records")

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"training set: {len(X_train)} records")
    print(f"test set: {len(X_test)} records")

    print("training xgboost model")
    model = train_model(X_train, y_train)

    print("evaluating model")
    metrics = evaluate_model(model, X_test, y_test)

    print("feature importance")
    feature_importance = get_feature_importance(model)

    save_model(model)

    return model, metrics, feature_importance


if __name__ == "__main__":
    run()