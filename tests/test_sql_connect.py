"""Unit tests for the SQL connection functions."""

from unittest.mock import MagicMock

import pandas as pd
import polars as pl

import databricks_warehouse.sql_connect as sc


def test_read_databricks_sql_connector(mocker, monkeypatch):
    """Use SQL connector when not in Databricks environment."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    mock_get_conn = mocker.patch("databricks_warehouse.sql_connect.get_sql_connection")
    conn = mock_get_conn.return_value.__enter__.return_value
    cursor = conn.cursor.return_value.__enter__.return_value
    expected_df = pd.DataFrame({"col": [1, 2]})
    cursor.fetchall_arrow.return_value.to_pandas.return_value = expected_df

    query = "SELECT *"
    params = {"p": 1}
    result = sc.read_databricks(query, host="h", cluster_id="c", params=params)

    mock_get_conn.assert_called_once_with(
        host="h", cluster_id="c", warehouse_id=None, client_id=None, client_secret=None
    )
    cursor.execute.assert_called_once_with(query, parameters=params)
    pd.testing.assert_frame_equal(result, expected_df)


def test_read_databricks_spark(mocker, monkeypatch):
    """Use Spark when in Databricks environment."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "1.0")
    mock_spark = MagicMock()
    mocker.patch("databricks_warehouse.sql_connect.get_spark_session", return_value=mock_spark)

    pdf = pd.DataFrame({"x": [9]})
    sdf = MagicMock(toPandas=MagicMock(return_value=pdf))
    mock_spark.sql.return_value = sdf

    result = sc.read_databricks("SELECT x", params={"x": 9})

    mock_spark.sql.assert_called_once_with("SELECT x", args={"x": 9})
    pd.testing.assert_frame_equal(result, pdf)


def test_read_databricks_pl_sql_connector(mocker, monkeypatch):
    """Use polars.read_database when not in Databricks environment."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    mock_get_conn = mocker.patch("databricks_warehouse.sql_connect.get_sql_connection")
    mock_read_db = mocker.patch("polars.read_database")
    conn = mock_get_conn.return_value.__enter__.return_value
    expected_pl = pl.DataFrame({"a": [3]})
    mock_read_db.return_value = expected_pl

    result = sc.read_databricks_pl(
        "SELECT *",
        warehouse_id="w",
        client_id="cid",
        client_secret="csec",
        schema_overrides={"a": "int"},
        params={"q": "v"},
    )

    mock_get_conn.assert_called_once_with(
        host=None, cluster_id=None, warehouse_id="w", client_id="cid", client_secret="csec"
    )
    mock_read_db.assert_called_once_with(
        query="SELECT *",
        connection=conn,
        schema_overrides={"a": "int"},
        execute_options={"parameters": {"q": "v"}},
    )
    assert result.equals(expected_pl)


def test_read_databricks_pl_spark(mocker, monkeypatch):
    """Use Spark + polars.from_pandas when in Databricks environment."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "runtime")
    mock_spark = MagicMock()
    mocker.patch("databricks_warehouse.sql_connect.get_spark_session", return_value=mock_spark)

    sdf = MagicMock()
    sdf.withColumn.return_value = sdf
    pd_df = pd.DataFrame({"y": [7]})
    sdf.toPandas.return_value = pd_df
    mock_spark.sql.return_value = sdf

    mock_from_pd = mocker.patch("polars.from_pandas")
    expected_pl = pl.DataFrame({"y": [7]})
    mock_from_pd.return_value = expected_pl

    result = sc.read_databricks_pl(
        "SELECT y",
        schema_overrides={"y": "long"},
        params={"y": 7},
    )

    mock_spark.sql.assert_called_once_with("SELECT y", args={"y": 7})
    sdf.withColumn.assert_called_once_with("y", sdf["y"].cast("long"))
    mock_from_pd.assert_called_once_with(pd_df)
    assert result.equals(expected_pl)


def test_execute_databricks_sql_connector(mocker, monkeypatch):
    """execute_databricks uses SQL connector when not in Databricks."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    mock_get_conn = mocker.patch("databricks_warehouse.sql_connect.get_sql_connection")
    conn = mock_get_conn.return_value.__enter__.return_value
    cursor = conn.cursor.return_value.__enter__.return_value

    result = sc.execute_databricks("DROP TABLE t", host="h", warehouse_id="w")

    mock_get_conn.assert_called_once_with(
        host="h", cluster_id=None, warehouse_id="w", client_id=None, client_secret=None
    )
    cursor.execute.assert_called_once_with("DROP TABLE t")
    assert result is None


def test_execute_databricks_spark(mocker, monkeypatch):
    """execute_databricks uses Spark when in Databricks."""
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "env")
    mock_spark = MagicMock()
    mocker.patch("databricks_warehouse.sql_connect.get_spark_session", return_value=mock_spark)

    result = sc.execute_databricks("ALTER TABLE t")

    mock_spark.sql.assert_called_once_with("ALTER TABLE t")
    assert result is None
