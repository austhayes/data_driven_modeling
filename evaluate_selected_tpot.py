import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one previously selected TPOT pipeline on the sealed test set."
    )
    parser.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    candidate = selection["selected_run"]
    pipeline = joblib.load(candidate["pipeline_path"])

    dataset = load_breast_cancer(as_frame=True)
    _, X_test, _, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        stratify=dataset.target,
        random_state=candidate["split_seed"],
    )
    predictions = pipeline.predict(X_test)
    score = balanced_accuracy_score(y_test, predictions)
    print(f"Held-out balanced accuracy: {score:.3f}")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=dataset.target_names,
        cmap="Blues",
    )
    plt.title(f"Selected TPOT pipeline (seed {candidate['search_seed']})")
    plt.show()


if __name__ == "__main__":
    main()
