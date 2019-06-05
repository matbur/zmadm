from pathlib import Path
import re
import io
import json
from pprint import pprint

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/{0}/{0}.data"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results")
DATASETS = (
    # name, label-column, categorical, ignore
    ("balance-scale", 4, (0,), ()),
    ("breast-cancer-wisconsin", 10, (), (0,)),
    ("ecoli", 8, (8,), (0,)),
    ("glass", 10, (), (0,)),
    ("haberman", 3, (), ()),
    ("iris", 4, (4,), ()),
    ("wine", 0, (), ()),
    ("letter-recognition", 0, (0,), ()),
)


def load_data(dataset: str, label_col: int, categorical=(), ignore=()) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / f"{dataset}.csv"

    if data_file.exists():
        df = pd.read_csv(data_file.as_posix(), header=None)

        for i in ignore:
            df = df.drop(i, axis=1)

        X = df.copy()
        y = X.pop(label_col)
        return X, y

    print(f"Downloading {dataset}")

    resp = requests.get(URL.format(dataset))
    resp.raise_for_status()

    text = re.sub(" +", ",", resp.text)
    buf = io.StringIO(text)
    df = pd.read_csv(buf, header=None)

    for i in categorical:
        df[i] = LabelEncoder().fit(df[i]).transform(df[i])

    if dataset == "breast-cancer-wisconsin":
        df = df[df[6] != "?"]
        df[6] = df[6].astype(int)

    df.to_csv(data_file.as_posix(), index=False, header=None)

    for i in ignore:
        df = df.drop(i, axis=1)

    X = df.copy()
    y = X.pop(label_col)
    return X, y


def get_score(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> int:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


def plot(dataset, scores):
    plot_file = RESULTS_DIR / f"{dataset}.png"

    bottom = min(scores.values()) ** 1.5 - 0.01
    heights = [i - bottom for i in scores.values()]
    print(bottom, heights)

    fig, ax = plt.subplots(1, 1)
    bar = ax.bar(scores.keys(), heights, bottom=bottom)
    ax.grid()
    ax.set_title(dataset)
    # label_bars(ax, bar, "{:.2f}")
    fig.savefig(plot_file.as_posix())


def label_bars(ax, bars, text_format, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart
    """
    ys = [bar.get_y() for bar in bars]
    y_is_constant = all(
        y == ys[0] for y in ys
    )  # -> regular bar chart, since all all bars start on the same y level (0)

    if y_is_constant:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_barh(ax, bars, text_format, **kwargs)


def _label_bar(ax, bars, text_format, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha="center", va="bottom", color=color, **kwargs)


def _label_barh(ax, bars, text_format, **kwargs):
    pass


def bagging_hyperparameters():
    for n_estimators in [5, 10, 20, 40, 100]:
        for max_samples in [0.3, 0.6, 1.0]:
            for max_features in [0.3, 0.6, 1.0]:
                for bootstrap in [True, False]:
                    for bootstrap_features in [True, False]:
                        yield {
                            "n_estimators": n_estimators,
                            "max_samples": max_samples,
                            "max_features": max_features,
                            "bootstrap": bootstrap,
                            "bootstrap_features": bootstrap_features,
                        }


def boosting_hyperparameters():
    for n_estimators in [5, 10, 20, 40, 100]:
        for learning_rate in [0.3, 0.6, 1.0]:
            for algorithm in ["SAMME", "SAMME.R"]:
                yield {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "algorithm": algorithm,
                }


def test_bagging(data):
    l = []
    for i in bagging_hyperparameters():
        print("running bagging", i)
        model = BaggingClassifier(**i)
        score = get_score(model, *data)
        print(score)
        l.append({"bagging": i, "score": score})
    return sorted(l, key=lambda x: x["score"], reverse=True)


def test_boosting(data):
    l = []
    for i in boosting_hyperparameters():
        print("running boosting", i)
        model = AdaBoostClassifier(**i)
        score = get_score(model, *data)
        print(score)
        l.append({"boosting": i, "score": score})
    return sorted(l, key=lambda x: x["score"], reverse=True)


def test_bagging_boosted(data, bagging, boosting):
    l = []
    for i in bagging:
        i = i["bagging"]
        for j in boosting:
            j = j["boosting"]
            print("running bagging boosted", i, j)
            model = BaggingClassifier(AdaBoostClassifier(**j), **i)
            score = get_score(model, *data)
            print(score)
            l.append({"bagging": i, "boosting": j, "score": score})
    return sorted(l, key=lambda x: x["score"], reverse=True)


def main():
    np.random.seed(42)
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}
    for i in DATASETS:
        dataset = i[0]
        print(dataset)
        X, y = load_data(*i)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        data = X_train, y_train, X_test, y_test

        n = 2 if dataset == "letter-recognition" else 5

        results_bagging = test_bagging(data)
        with open(RESULTS_DIR / f"{dataset}_bagging.json", "w") as f:
            json.dump(results_bagging, f, indent="  ")
        best_bagging = results_bagging[0]

        result_boosting = test_boosting(data)
        with open(RESULTS_DIR / f"{dataset}_boosting.json", "w") as f:
            json.dump(result_boosting, f, indent="  ")
        best_boosting = result_boosting[0]

        results_bagging_boosted = test_bagging_boosted(
            data, results_bagging[:n], result_boosting[:n]
        )
        with open(RESULTS_DIR / f"{dataset}_bagging_boosted.json", "w") as f:
            json.dump(results_bagging_boosted, f, indent="  ")
        best_bagging_boosted = results_bagging_boosted[0]

        scores = {
            "bagging": best_bagging["score"],
            "adaboost": best_boosting["score"],
            "bagging_adaboost": best_bagging_boosted["score"],
        }

        for method, score in scores.items():
            print(method, score)

        plot(dataset, scores)
        results[dataset] = scores

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent="  ")


if __name__ == "__main__":
    main()
