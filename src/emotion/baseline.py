from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

from .data import prepare_goemotions_mood_dataset
from .labels import ID_TO_MOOD


def train_emotion_baseline(
    cache_dir: str | None = None,
    model_dir: str = "models/emotion_baseline",
    max_features: int = 20000,
    C: float = 2.0,
) -> None:
    train_df, val_df, test_df = prepare_goemotions_mood_dataset(cache_dir=cache_dir)

    X_train, y_train = train_df["text"].tolist(), train_df["mood_id"].to_numpy()
    X_val, y_val = val_df["text"].tolist(), val_df["mood_id"].to_numpy()
    X_test, y_test = test_df["text"].tolist(), test_df["mood_id"].to_numpy()

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    multi_class="multinomial",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")

    print("Validation accuracy:", val_acc)
    print("Validation macro F1:", val_f1_macro)
    print("Validation classification report:")
    print(
        classification_report(
            y_val,
            y_val_pred,
            target_names=[ID_TO_MOOD[i] for i in sorted(ID_TO_MOOD.keys())],
        )
    )

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    print("Test accuracy:", test_acc)
    print("Test macro F1:", test_f1_macro)
    print("Test classification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=[ID_TO_MOOD[i] for i in sorted(ID_TO_MOOD.keys())],
        )
    )

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path / "emotion_baseline.joblib")

    print(f"Baseline model saved to {model_path / 'emotion_baseline.joblib'}")


def load_emotion_baseline(model_dir: str = "models/emotion_baseline") -> Pipeline:
    model_path = Path(model_dir) / "emotion_baseline.joblib"
    clf = joblib.load(model_path)
    return clf


def predict_mood_baseline(clf: Pipeline, text: str) -> dict:
    pred_id = int(clf.predict([text])[0])
    mood = ID_TO_MOOD[pred_id]
    return {"mood_id": pred_id, "mood": mood}

if __name__ == "__main__":
    train_emotion_baseline()
