"""A wrapper for the databricks-sql-connector to run queries that return Pandas or Polars DataFrames."""

import os
from typing import Optional

import databricks.sql
import databricks.sql.client
import pandas as pd
import polars as pl
from databricks.sdk.config import Config
from databricks.sdk.credentials_provider import databricks_cli, oauth_service_principal


def read_databricks(
    query: str,
    host: Optional[str] = None,
    cluster_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    params: dict = {},
) -> pd.DataFrame:
    """Execute a query on a Databricks cluster or warehouse and return the result as a Pandas DataFrame.

    This is the easiest way to run queries that return immediate results and do not require PySpark.
    The target can be either a SQL warehouse or general purpose compute.

    All arguments (except query) are optional and can be used to override settings from the environment or a configuration file.

    Args:
        query: The SQL query to run.
        host: The host of the Databricks workspace.
        cluster_id: The ID of the compute cluster to run the query on.  Cannot be set with warehouse_id.
        warehouse_id: The ID of the SQL warehouse to run the query on.  Cannot be set with cluster_id.
        client_id: The client ID for M2M OAuth. Cannot be set with token.
        client_secret: The client secret for M2M OAuth. Cannot be set with token.
        params: SQL query parameters to pass in to the query -- use `:name` as the parameter placeholder.

    If client_id and client_secret are not provided, the Databricks CLI will be used for authentication.

    Returns:
        A Pandas DataFrame with the results of the query
    """
    if running_in_databricks():
        spark = get_spark_session()
        return spark.sql(query, args=params).toPandas()
    else:
        with get_sql_connection(
            host=host,
            cluster_id=cluster_id,
            warehouse_id=warehouse_id,
            client_id=client_id,
            client_secret=client_secret,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, parameters=params)
                return cursor.fetchall_arrow().to_pandas()


def read_databricks_pl(
    query: str,
    host: Optional[str] = None,
    cluster_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    schema_overrides: Optional[dict] = None,
    params: dict = {},
) -> pl.DataFrame:
    """Execute a query on a Databricks cluster or warehouse and return the result as a Polars DataFrame.

    All arguments (except query and kwargs) are optional and can be used to override settings
    from the environment or a configuration file.

    Args:
        query: The SQL query to run.
        host: The host of the Databricks workspace.
        cluster_id: The ID of the compute cluster to run the query on.  Cannot be set with warehouse_id.
        warehouse_id: The ID of the SQL warehouse to run the query on.  Cannot be set with cluster_id.
        client_id: The client ID for M2M OAuth. Cannot be set with token.
        client_secret: The client secret for M2M OAuth. Cannot be set with token.
        schema_overrides: Dictionary mapping columns to intended dtypes.
        params: SQL query parameters to pass in to the query -- use `:name` as the parameter placeholder.

    If client_id and client_secret are not provided, the Databricks CLI will be used for authentication.

    Returns:
        A Polars DataFrame with the results of the query
    """
    if running_in_databricks():
        spark = get_spark_session()
        spark_df = spark.sql(query, args=params)
        if schema_overrides:
            # Apply schema overrides to the Spark DataFrame before converting to Pandas
            for col, dtype in schema_overrides.items():
                spark_df = spark_df.withColumn(col, spark_df[col].cast(dtype))

        # Convert to Polars using Pandas as an intermediate step
        # Using Arrow would be more efficient, but Spark Connect does not support it
        return pl.from_pandas(spark_df.toPandas())
    else:
        with get_sql_connection(
            host=host,
            cluster_id=cluster_id,
            warehouse_id=warehouse_id,
            client_id=client_id,
            client_secret=client_secret,
        ) as conn:
            return pl.read_database(
                query=query,
                connection=conn,
                schema_overrides=schema_overrides,
                execute_options={"parameters": params},
            )


def execute_databricks(
    stmt: str,
    host: Optional[str] = None,
    cluster_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> None:
    """Execute a SQL statement on a Databricks cluster or warehouse.

    This is primarily useful for non-SELECT statements which do not return any results.

    All arguments (except stmt) are optional and can be used to override settings from the environment or a configuration file.

    Args:
        stmt: The SQL statement to run.
        host: The host of the Databricks workspace.
        cluster_id: The ID of the compute cluster to run the query on.  Cannot be set with warehouse_id.
        warehouse_id: The ID of the SQL warehouse to run the query on.  Cannot be set with cluster_id.
        client_id: The client ID for M2M OAuth.
        client_secret: The client secret for M2M OAuth.

    If client_id and client_secret are not provided, the Databricks CLI will be used for authentication.
    """
    if running_in_databricks():
        spark = get_spark_session()
        spark.sql(stmt)
    else:
        with get_sql_connection(
            host=host,
            cluster_id=cluster_id,
            warehouse_id=warehouse_id,
            client_id=client_id,
            client_secret=client_secret,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(stmt)


def get_sql_connection(
    host: Optional[str] = None,
    cluster_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> databricks.sql.client.Connection:
    """Return a connection to the Databricks SQL endpoint.

    Args:
        host: The host of the Databricks workspace.
        cluster_id: The ID of the compute cluster to run the query on.  Cannot be set with warehouse_id.
        warehouse_id: The ID of the SQL warehouse to run the query on.  Cannot be set with cluster_id.
        client_id: The client ID for M2M OAuth.
        client_secret: The client secret for M2M OAuth.

    If client_id and client_secret are not provided, the Databricks CLI will be used for authentication.
    """
    config = Config(
        host=host,
        cluster_id=cluster_id,
        warehouse_id=warehouse_id,
        client_id=client_id,
        client_secret=client_secret,
    )

    if config.sql_http_path is None:
        raise ValueError(
            "Unable to resolve HTTP path based on provided settings.\n"
            "Make sure you're providing a cluster ID or warehouse ID, either from "
            "$DATABRICKS_CLUSTER_ID or $DATABRICKS_WAREHOUSE_ID environment variables, or a ~/.databrickscfg file."
        )

    # We are intentionally choosing to only support M2M and U2M OAuth, not Personal Access Token
    credentials = oauth_service_principal(config)
    if credentials is None:
        credentials = databricks_cli(config)
    if credentials is None:
        raise ValueError(
            "Unable to generate OAuth credentials for Databricks SQL connection using either a "
            "Service Principal (M2M) or the Databricks CLI (U2M). "
            "For M2M OAuth, make sure $DATABRICKS_CLIENT_ID and $DATABRICKS_CLIENT_SECRET are provided "
            "as environment variables or a ~/.databrickscfg file."
            "\n\nFor the U2M OAuth, make sure the Databricks CLI is installed and run\n"
            f"databricks auth login --host {config.host}\nto generate a token."
        )

    return databricks.sql.connect(
        server_hostname=config.hostname,
        http_path=config.sql_http_path,
        credentials_provider=lambda: credentials,
    )


def running_in_databricks() -> bool:
    """Returns whether the code is running on a Databricks compute cluster."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark_session():
    try:
        from databricks.connect import DatabricksSession
    except ImportError:
        raise ImportError(
            "Databricks Connect is not installed.  To include it, make sure you're adding the `databricks_warehouse` "
            "as a dependecy with the `connect` extra, e.g. `pip install databricks_warehouse[connect]` or "
            "`poetry add databricks_warehouse --extras connect'"
        )
    return DatabricksSession.builder.getOrCreate()
