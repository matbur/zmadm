#/usr/bin/env python

from pathlib import Path
import re
import math
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
    # name, label-column, categorical, ignore, target
    ("balance-scale", 0, (0, ), (), ''),
    ("breast-cancer-wisconsin", 10, (), (0, ), ''),
    ("ecoli", 8, (8, ), (0, ), ''),
    ("glass", 10, (), (0, ), ''),
    ("haberman", 3, (), (), ''),
    ("iris", 4, (4, ), (), ''),
    ("wine", 0, (), (), ''),
    ("letter-recognition", 0, (0, ), (), ''),
    ("abalone", 0, (0, ), (), 'sex'),
    ('cmc', 9, (), (), 'method'),
    ('cmc', 4, (), (), 'islam'),
    ('cmc', 8, (), (), 'media'),
    ('cmc', 5, (), (), 'working'),
    ('car', 6, (0, 1, 2, 3, 4, 5, 6), (), ''),
    ('hayes-roth', 5, (), (), ''),
    ('tae', 5, (), (), ''),
    ('lymphography', 18, (), (), ''),
    ('winequality-red', 11, (), (), ''),
    ('heart', 13, (), (), ''),
    ('pulsar_stars', 8, (), (), ''),
    ('soybean-small', 35, (), (), ''),
    ('flag', 28, (17, 28, 29), (0, ), ''),
    # ('', 0, (), (), ''),
)


def load_data(dataset: str, label_col: int, categorical=(),
              ignore=()) -> pd.DataFrame:
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

    bottom = min(scores.values())**1.5 - 0.01
    heights = [i - bottom for i in scores.values()]
    # print(bottom, heights)

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

        ax.text(
            text_x,
            text_y,
            text,
            ha="center",
            va="bottom",
            color=color,
            **kwargs)


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
        # print("running bagging", i)
        model = BaggingClassifier(**i)
        score = get_score(model, *data)
        # print(score)
        l.append({"bagging": i, "score": score})
    return sorted(l, key=lambda x: x["score"], reverse=True)


def test_boosting(data):
    l = []
    for i in boosting_hyperparameters():
        # print("running boosting", i)
        model = AdaBoostClassifier(**i)
        score = get_score(model, *data)
        # print(score)
        l.append({"boosting": i, "score": score})
    return sorted(l, key=lambda x: x["score"], reverse=True)


def main():
    np.random.seed(42)
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}
    bests_bagging = {}
    bests_boosting = {}
    for i in DATASETS:
        dataset = f'{i[0]}_{i[-1]}' if i[-1] != '' else i[0]
        print(dataset)
        X, y = load_data(*i[:-1])
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        data = X_train, y_train, X_test, y_test

        results_bagging = test_bagging(data)
        with open(RESULTS_DIR / f"{dataset}_bagging.json", "w") as f:
            json.dump(results_bagging, f, indent="  ")
        best_bagging = results_bagging[0]

        result_boosting = test_boosting(data)
        with open(RESULTS_DIR / f"{dataset}_boosting.json", "w") as f:
            json.dump(result_boosting, f, indent="  ")
        best_boosting = result_boosting[0]

        scores = {
            "bagging": best_bagging["score"],
            "adaboost": best_boosting["score"],
        }

        for method, score in scores.items():
            print(method, score)

        plot(dataset, scores)
        results[dataset] = scores
        bests_bagging[dataset] = best_bagging
        bests_boosting[dataset] = best_boosting

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent="  ")

    with open(RESULTS_DIR / "bests_bagging.json", "w") as f:
        json.dump(bests_bagging, f, indent="  ")

    with open(RESULTS_DIR / 'bagging.txt', 'w') as f:
        for i, (k, v) in enumerate(bests_bagging.items(), 1):
            f.write('{} & {} & {:.3f} \\\\ \\hline \n'.format(
                i, ' & '.join([str(i) for i in v['bagging'].values()]),
                v['score']))

    with open(RESULTS_DIR / "bests_boosting.json", "w") as f:
        json.dump(bests_boosting, f, indent="  ")

    with open(RESULTS_DIR / 'boosting.txt', 'w') as f:
        for i, (k, v) in enumerate(bests_boosting.items(), 1):
            f.write('{} & {} & {:.3f} \\\\ \\hline \n'.format(
                i, ' & '.join([str(i) for i in v['boosting'].values()]),
                v['score']))


def images():
    TEMPLATE = r"""\begin{{figure}}[h!]
\item{{{}}}
\begin{{center}}
\includegraphics[width=250pt]{{{}}}
\caption{{Dokładność dla bazy {}}}
\end{{center}}
\end{{figure}}

"""
    with open(RESULTS_DIR / 'plots.txt', 'w') as f:
        for i in sorted(RESULTS_DIR.glob('*.png')):
            name = i.stem.replace('-', ' ').replace('_', ' ')
            f.write(
                TEMPLATE.format(
                    name,
                    i.as_posix().replace(RESULTS_DIR.as_posix(), 'img'), name))


def datasets():
    with open(RESULTS_DIR / 'datasets.txt', 'w') as f:
        for i in DATASETS:
            dataset = f'{i[0]}_{i[-1]}' if i[-1] != '' else i[0]
            dataset = dataset.replace('-', ' ').replace('_', ' ')
            X, y = load_data(*i[:-1])
            ylen = len(np.unique(y))
            f.write('{} & {} & {} & {} \\\\ \\hline \n'.format(
                dataset, X.shape[0], X.shape[1], ylen))


def wilcoxon():
    with open(RESULTS_DIR / 'bests_bagging.json') as f:
        bests_bagging = json.load(f)

    with open(RESULTS_DIR / 'bests_boosting.json') as f:
        bests_boosting = json.load(f)

    def foo(ba, bo):
        if math.isclose(ba, bo):
            return 'R & R'
        if ba > bo:
            return '+ & -'
        if bo > ba:
            return '- & +'

        exit(42)

    with open(RESULTS_DIR / 'wilcoxon.txt', 'w') as f:
        for i in DATASETS:
            dataset = f'{i[0]}_{i[-1]}' if i[-1] != '' else i[0]
            bagging = bests_bagging[dataset]['score']
            boosting = bests_boosting[dataset]['score']
            dataset = dataset.replace('-', ' ').replace('_', ' ')
            f.write(
                '\multicolumn{{1}}{{|l|}}{{{}}} & {:.03f} & {:.3f} & {} \\\\ \hline \n'
                .format(dataset, bagging, boosting, foo(bagging, boosting)))


if __name__ == "__main__":
    main()
    images()
    datasets()
    wilcoxon()