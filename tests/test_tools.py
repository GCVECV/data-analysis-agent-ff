# tests/test_tools.py
import pandas as pd
import pytest
from core.datasource import CSVDataSource
from core.tools_csv import data_clean, group_aggregate, calculate_stats, plot_chart

@pytest.fixture
def ds():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B", "C"],
        "sales": [10, 20, 30, 40, 500],   # 500 是异常值
        "price": [1.0, None, 2.0, 3.0, 4.0],
    })
    return CSVDataSource(df)


def test_group_aggregate_ok(ds):
    r = group_aggregate({"group_column": "region", "agg_column": "sales", "agg_func": "sum"}, ds)
    assert r.ok
    assert r.data["result"]["C"] == 500


def test_group_aggregate_bad_column(ds):
    r = group_aggregate({"group_column": "不存在", "agg_column": "sales", "agg_func": "sum"}, ds)
    assert not r.ok
    assert "可用列" in r.text


def test_group_aggregate_non_numeric(ds):
    r = group_aggregate({"group_column": "region", "agg_column": "region", "agg_func": "sum"}, ds)
    assert not r.ok


def test_remove_outliers(ds):
    r = data_clean({"action": "remove_outliers", "column": "sales"}, ds)
    assert r.ok
    assert r.data["removed"] == 1
    assert len(ds.get_dataframe()) == 4


def test_fill_na_mean(ds):
    r = data_clean({"action": "fill_na", "column": "price", "fill_value": "mean"}, ds)
    assert r.ok
    assert ds.get_dataframe()["price"].isna().sum() == 0


def test_plot_bar_non_numeric_y(ds):
    r = plot_chart({"chart_type": "bar", "x_column": "region", "y_column": "region"}, ds)
    assert not r.ok
    assert "非数值" in r.error


def test_stats_ok(ds):
    r = calculate_stats({"column": "sales"}, ds)
    assert r.ok
    assert r.data["max"] == 500


def test_group_aggregate_month_granularity():
    """按月聚合：日期分组列转为 YYYY-MM 格式。"""
    df = pd.DataFrame({
        "date": ["2024-01-05", "2024-01-20", "2024-02-01", "2024-02-10"],
        "sales": [10, 20, 30, 40],
    })
    d = CSVDataSource(df)
    r = group_aggregate({"group_column": "date", "agg_column": "sales", "agg_func": "sum", "granularity": "month"}, d)
    assert r.ok
    keys = set(r.data["result"].keys())
    assert keys == {"2024-01", "2024-02"}
    assert r.data["result"]["2024-01"] == 30


def test_plot_chart_reuses_aggregation(ds):
    """绘图复用最近聚合结果：聚合后 y 列无需存在。"""
    r = group_aggregate({"group_column": "region", "agg_column": "sales", "agg_func": "sum"}, ds)
    assert r.ok
    p = plot_chart({"chart_type": "bar", "x_column": "region", "y_column": "sales"}, ds)
    assert p.ok
    assert p.artifacts and p.artifacts[0]["image"]


def test_plot_chart_line_from_month_agg():
    """月度聚合 + 折线图：直接用聚合结果绘图。"""
    df = pd.DataFrame({
        "date": ["2024-01-05", "2024-02-20", "2024-03-01"],
        "sales": [10, 20, 30],
    })
    d = CSVDataSource(df)
    r = group_aggregate({"group_column": "date", "agg_column": "sales", "agg_func": "sum", "granularity": "month"}, d)
    assert r.ok
    p = plot_chart({"chart_type": "line", "x_column": "date", "y_column": "sales"}, d)
    assert p.ok
    assert p.artifacts