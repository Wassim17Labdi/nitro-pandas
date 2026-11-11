"""Runs the same pandas code on pandas and nitro-pandas and compares the results."""
import datetime as dt
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import polars as pl

import nitro_pandas as npd

DATA = {
    "k": ["b", "a", "b", None, "a"],
    "v": [1.0, None, 3.0, 4.0, 5.0],
    "n": [1, 2, 3, 4, 5],
    "s": ["Hello a", "world", "a a", None, "x"],
    "d": [dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 3), dt.datetime(2024, 3, 6), dt.datetime(2024, 4, 9), dt.datetime(2024, 5, 12)],
}
RIGHT = {"k": ["a", "b", None], "v": [10.0, 20.0, 30.0], "extra": [7, 8, 9]}
LOOKUP = {"code": [1, 2], "label": ["x", "y"]}


def _right(L, data=RIGHT):
    return npd.DataFrame(pl.DataFrame(data)) if L is npd else pl.DataFrame(data).to_pandas()


def _counts(result):
    if isinstance(result, npd.DataFrame):
        frame = result.to_pandas()
        return sorted(zip(frame.iloc[:, 0], frame["count"]))
    return sorted(result.items())


def _mutated(fn):
    def run(L, d):
        fn(L, d)
        return d
    return run


CASES = {
    "len": lambda L, d: len(d),
    "attribute access": lambda L, d: d.n,
    "iterate columns": lambda L, d: list(d),
    "contains column": lambda L, d: "n" in d,
    "add columns": lambda L, d: d["n"] + d["v"],
    "negate": lambda L, d: -d["n"],
    "positional item": lambda L, d: d["n"][1],
    "invert mask keeps nulls": lambda L, d: d[~(d["v"] > 2)],
    "not equal keeps nulls": lambda L, d: d[d["v"] != 3],
    "mask sum": lambda L, d: (d["v"] > 2).sum(),
    "series head default": lambda L, d: d["n"].head(),
    "series clip": lambda L, d: d["n"].clip(lower=2),
    "series round": lambda L, d: (d["n"] / 8).round(2),
    "series stats skip nulls": lambda L, d: [d["v"].sum(), d["v"].mean(), d["v"].count(), d["v"].nunique()],
    "series quantile": lambda L, d: d["v"].quantile(0.25),
    "series astype str": lambda L, d: d["v"].astype(str),
    "series value_counts drops nulls": lambda L, d: _counts(d["k"].value_counts()),
    "str lower": lambda L, d: d["s"].str.lower(),
    "str upper": lambda L, d: d["s"].str.upper(),
    "str len": lambda L, d: d["s"].str.len(),
    "str strip": lambda L, d: d["s"].str.strip(),
    "str startswith": lambda L, d: d["s"].str.startswith("a"),
    "str endswith": lambda L, d: d["s"].str.endswith("a"),
    "str slice": lambda L, d: d["s"].str[:3],
    "str replace all": lambda L, d: d["s"].str.replace("a", "_"),
    "str replace regex": lambda L, d: d["s"].str.replace("[ao]", "_", regex=True),
    "str replace literal dollar": lambda L, d: d["s"].str.replace("a", "$1"),
    "str contains ignore case": lambda L, d: d["s"].str.contains("HELLO", case=False, na=False),
    "str contains lookahead": lambda L, d: d["s"].str.contains("a(?= a)", na=False),
    "dt year": lambda L, d: d["d"].dt.year,
    "dt dayofweek": lambda L, d: d["d"].dt.dayofweek,
    "dt strftime": lambda L, d: d["d"].dt.strftime("%Y-%m"),
    "assign from dt": _mutated(lambda L, d: d.__setitem__("y", d["d"].dt.month)),
    "drop columns": lambda L, d: d.drop(columns=["s"]),
    "drop_duplicates keeps order": lambda L, d: d.drop_duplicates(subset=["k"]),
    "duplicated": lambda L, d: d.duplicated(subset=["k"]),
    "dropna": lambda L, d: d.dropna(),
    "dropna subset": lambda L, d: d.dropna(subset=["v"]),
    "fillna dict": lambda L, d: d.fillna({"v": 0}),
    "sort multiple": lambda L, d: d.sort_values(["k", "n"], ascending=[False, True]),
    "isna sum": lambda L, d: d.isna().sum(),
    "frame mean": lambda L, d: d.mean(numeric_only=True),
    "frame sum": lambda L, d: d.sum(numeric_only=True),
    "frame median": lambda L, d: d.median(numeric_only=True),
    "frame std": lambda L, d: d.std(numeric_only=True),
    "frame count": lambda L, d: d.count(),
    "frame nunique": lambda L, d: d.nunique(),
    "frame corr": lambda L, d: d[["v", "n"]].corr(),
    "frame describe": lambda L, d: d[["v", "n"]].describe(),
    "frame copy": lambda L, d: d.copy(),
    "frame assign": lambda L, d: d.assign(z=lambda f: f["n"] * 2),
    "reset_index adds row numbers": lambda L, d: d[["n"]].reset_index(),
    "inplace fillna": _mutated(lambda L, d: d.fillna({"v": 0}, inplace=True)),
    "inplace dropna": _mutated(lambda L, d: d.dropna(inplace=True)),
    "inplace drop": _mutated(lambda L, d: d.drop(columns=["s"], inplace=True)),
    "inplace rename": _mutated(lambda L, d: d.rename(columns={"n": "m"}, inplace=True)),
    "inplace sort": _mutated(lambda L, d: d.sort_values("n", ascending=False, inplace=True)),
    "del column": _mutated(lambda L, d: d.__delitem__("s")),
    "insert column": _mutated(lambda L, d: d.insert(1, "z", d["n"] * 2)),
    "pop column": _mutated(lambda L, d: d.pop("s")),
    "pending column reaches drop": _mutated(lambda L, d: (d.__setitem__("z", 1), d.drop(columns=["s"], inplace=True))),
    "pending column reaches rename": lambda L, d: (d.__setitem__("z", 1), d.rename(columns={"z": "w"})[["w"]])[1],
    "loc assign": _mutated(lambda L, d: d.loc.__setitem__((d["n"] > 3, "v"), 0.0)),
    "loc assign new column": _mutated(lambda L, d: d.loc.__setitem__((d["n"] > 3, "z"), 1.0)),
    "groupby sorted keys, null keys dropped": lambda L, d: d.groupby("k")["n"].mean(),
    "groupby keeps null keys on request": lambda L, d: d.groupby("k", dropna=False)["n"].sum(),
    "groupby first appearance": lambda L, d: d.groupby("k", sort=False)["n"].sum(),
    "groupby as_index": lambda L, d: d.groupby("k", as_index=False)["n"].sum(),
    "groupby count per column": lambda L, d: d[["k", "v", "n"]].groupby("k").count(),
    "groupby size": lambda L, d: d.groupby("k").size(),
    "groupby first skips nulls": lambda L, d: d[["k", "v"]].groupby("k").first(),
    "groupby nunique": lambda L, d: d.groupby("k")["n"].nunique(),
    "groupby std": lambda L, d: d.groupby("k")["n"].std(),
    "groupby median": lambda L, d: d.groupby("k")["n"].median(),
    "groupby transform": lambda L, d: d.groupby("k")["n"].transform("sum"),
    "groupby agg dict": lambda L, d: d.groupby("k").agg({"n": "median", "s": "first"}),
    "groupby named agg": lambda L, d: d.groupby("k").agg(a=("n", "sum"), b=("v", "max")),
    "groupby iterate keys": lambda L, d: [key for key, _ in d.groupby("k")],
    "groupby reset_index": lambda L, d: d.groupby("k")["n"].sum().reset_index(),
    "groupby sort result": lambda L, d: d.groupby("k")["n"].sum().sort_values(ascending=False),
    "merge suffixes": lambda L, d: d[["k", "v"]].merge(_right(L), on="k"),
    "merge null keys": lambda L, d: d[["k", "n"]].merge(_right(L), on="k", how="left"),
    "merge outer one key column": lambda L, d: d[["k", "n"]].merge(_right(L), on="k", how="outer"),
    "merge keeps both named keys": lambda L, d: d[["n"]].merge(_right(L, LOOKUP), left_on="n", right_on="code"),
    "module merge": lambda L, d: L.merge(d[["k", "n"]], _right(L), on="k"),
    "concat ignore_index": lambda L, d: L.concat([d[["n"]], d[["n"]]], ignore_index=True),
    "concat union of columns": lambda L, d: L.concat([d[["n"]], d[["v"]]]),
    "concat series": lambda L, d: L.concat([d[["n"]], d["v"]], axis=1),
    "cut": lambda L, d: L.cut(d["n"], 2).astype(str),
    "get_dummies": lambda L, d: L.get_dummies(d[["k"]]),
    "pivot_table drops empty columns": lambda L, d: d.pivot_table(values="v", index="k", columns="n", aggfunc="sum"),
}


def _plain(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return None if number != number else round(number, 6)
    return str(value)


def _normalise(result):
    if isinstance(result, (npd.DataFrame, npd.dataframe.Series, pl.Series)):
        result = result.to_pandas()
    if isinstance(result, pd.Series):
        result = result.reset_index() if result.index.names != [None] or isinstance(result.index[:1][0] if len(result) else 0, str) else result.to_frame()
    if isinstance(result, pd.DataFrame):
        if result.index.names != [None] or isinstance(result.index[:1][0] if len(result) else 0, str):
            result = result.reset_index()
        return [tuple(_plain(v) for v in row) for row in result.astype(object).to_numpy().tolist()]
    if isinstance(result, (list, tuple, np.ndarray)):
        return [_normalise(v) for v in result]
    return _plain(result)


def test_pandas_parity():
    failures = []
    for name, case in CASES.items():
        frame = pl.DataFrame(DATA)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = case(pd, frame.to_pandas())
            try:
                actual = _normalise(case(npd, npd.DataFrame(frame)))
            except Exception as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
                continue
        if actual != _normalise(expected):
            failures.append(f"{name}: {actual} != {_normalise(expected)}")
    assert not failures, "\n".join(failures)
    print(f"  OK test_pandas_parity ({len(CASES)} cases)")


def test_series_truth_value_is_ambiguous():
    try:
        bool(npd.DataFrame({"a": [1, 2]})["a"] > 1)
    except ValueError:
        print("  OK test_series_truth_value_is_ambiguous")
        return
    assert False, "A Series must not have a truth value"


def test_read_csv_matches_pandas():
    content = "code,day,val,txt\n0012,2024-01-05,1.5,x\n0034,2024-02-06,NA,y\n0056,2024-03-07,2.5,null\n"
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "data.csv")
        with open(path, "w") as handle:
            handle.write(content)
        options = [{}, {"dtype": {"code": str}}, {"dtype": {"code": "str"}}, {"usecols": ["code", "val"]}, {"parse_dates": ["day"]}]
        for kwargs in options:
            actual, expected = npd.read_csv(path, **kwargs), pd.read_csv(path, **kwargs)
            assert _normalise(actual) == _normalise(expected), f"read_csv({kwargs})"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert isinstance(npd.read_csv(path, index_col=0), pd.DataFrame), "pandas-only options go through pandas"
    print("  OK test_read_csv_matches_pandas")


def test_to_csv_matches_pandas():
    frame = npd.DataFrame({"a": [1, 2], "b": ["x", None]})
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "out.csv")
        frame.to_csv(path, index=False, sep=";", na_rep="-")
        with open(path) as handle:
            assert handle.read() == frame.to_pandas().to_csv(index=False, sep=";", na_rep="-")
    print("  OK test_to_csv_matches_pandas")


if __name__ == "__main__":
    test_pandas_parity()
    test_series_truth_value_is_ambiguous()
    test_read_csv_matches_pandas()
    test_to_csv_matches_pandas()
