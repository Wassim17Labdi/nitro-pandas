"""Tests pour nitro_pandas.profiling (profile_compare)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nitro_pandas as npd
from nitro_pandas.profiling import ProfileComparison, ProfileRow


# ── workloads de test ────────────────────────────────────────────────────────

def _simple_workload(pd):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    result = df[df["a"] > 2]
    return result


def _groupby_workload(pd):
    df = pd.DataFrame({
        "cat": ["x", "y", "x", "y", "x"],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    return df.groupby("cat")["val"].mean()


def _fallback_workload(pd):
    df = pd.DataFrame({"a": [3.0, 1.0, 2.0], "b": [30.0, 10.0, 20.0]})
    # .cumsum() n'est pas implémenté nativement → déclenche un fallback
    return df.cumsum()


# ── tests ────────────────────────────────────────────────────────────────────

def test_profile_compare_returns_comparison():
    result = npd.profile_compare(_simple_workload)
    assert isinstance(result, ProfileComparison), "Doit retourner un ProfileComparison"
    assert len(result.rows) > 0, "Doit contenir au moins une ligne"
    assert result.pandas_total > 0, "Temps pandas doit être > 0"
    assert result.nitro_total > 0, "Temps nitro doit être > 0"
    assert result.gain_total > 0, "Gain total doit être > 0"
    print("  OK test_profile_compare_returns_comparison")


def test_profile_compare_rows_structure():
    result = npd.profile_compare(_simple_workload)
    for row in result.rows:
        assert isinstance(row, ProfileRow)
        assert isinstance(row.lineno, int)
        assert isinstance(row.source, str)
        assert row.pandas_s >= 0
        assert row.nitro_s >= 0
        assert isinstance(row.fell_back, bool)
    print("  OK test_profile_compare_rows_structure")


def test_profile_compare_str():
    result = npd.profile_compare(_simple_workload)
    output = str(result)
    assert "pandas" in output
    assert "nitro" in output
    assert "Gain" in output
    assert "TOTAL" in output
    print("  OK test_profile_compare_str")


def test_profile_compare_return_dict():
    result = npd.profile_compare(_simple_workload, return_format="dict")
    assert isinstance(result, list)
    assert len(result) > 0
    keys = {"lineno", "source", "pandas_s", "nitro_s", "gain", "fell_back"}
    assert keys == result[0].keys()
    print("  OK test_profile_compare_return_dict")


def test_profile_compare_return_dataframe():
    result = npd.profile_compare(_simple_workload, return_format="dataframe")
    assert isinstance(result, npd.DataFrame)
    pd_df = result.to_pandas()
    assert "lineno" in pd_df.columns
    assert "gain" in pd_df.columns
    assert "fell_back" in pd_df.columns
    print("  OK test_profile_compare_return_dataframe")


def test_profile_compare_assert_equal_passes():
    # Les deux backends doivent retourner des valeurs équivalentes
    npd.profile_compare(_groupby_workload, assert_equal=True)
    print("  OK test_profile_compare_assert_equal_passes")


def test_profile_compare_assert_equal_fails():
    def divergent_workload(pd):
        # nitro_pandas.__name__ == "nitro_pandas", pandas.__name__ == "pandas"
        is_nitro = pd.__name__ == "nitro_pandas"
        values = [10, 20, 30] if is_nitro else [1, 2, 3]
        return pd.DataFrame({"a": values})

    raised = False
    try:
        npd.profile_compare(divergent_workload, assert_equal=True)
    except AssertionError:
        raised = True
    assert raised, "assert_equal doit lever AssertionError si les résultats diffèrent"
    print("  OK test_profile_compare_assert_equal_fails")


def test_profile_compare_n_runs():
    result = npd.profile_compare(_simple_workload, n_runs=3)
    assert isinstance(result, ProfileComparison)
    assert result.pandas_total > 0
    print("  OK test_profile_compare_n_runs")


def test_profile_compare_warmup():
    result = npd.profile_compare(_simple_workload, warmup=2)
    assert isinstance(result, ProfileComparison)
    print("  OK test_profile_compare_warmup")


def test_profile_compare_fallback_detected():
    result = npd.profile_compare(_fallback_workload)
    # Au moins une ligne doit être marquée comme fallback
    assert any(r.fell_back for r in result.rows), (
        "Une ligne doit être marquée fell_back=True quand un fallback pandas est déclenché"
    )
    print("  OK test_profile_compare_fallback_detected")


def test_profile_compare_to_dataframe():
    result = npd.profile_compare(_simple_workload)
    df = result.to_dataframe()
    assert isinstance(df, npd.DataFrame)
    assert df.shape[0] == len(result.rows)
    print("  OK test_profile_compare_to_dataframe")


if __name__ == "__main__":
    test_profile_compare_returns_comparison()
    test_profile_compare_rows_structure()
    test_profile_compare_str()
    test_profile_compare_return_dict()
    test_profile_compare_return_dataframe()
    test_profile_compare_assert_equal_passes()
    test_profile_compare_assert_equal_fails()
    test_profile_compare_n_runs()
    test_profile_compare_warmup()
    test_profile_compare_fallback_detected()
    test_profile_compare_to_dataframe()
    print("\nOK Tous les tests profiling sont passés !")
