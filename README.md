# Databricks Warehouse

A Python library that wraps the Databricks SQL connector (with optional fallbacks to Databricks Connect) and provides a standard interface for querying SQL warehouses from outside of Databricks.


# Overview

This library offers a simplified interface for running SQL queries on a Databricks cluster or warehouse and handling authentication using OAuth.

The goal of this library is to make querying Databricks "just work" in a standard way across projects for Data Scientists.  This library is ideal for projects like external apps that sit outside of Databricks, run relatively small queries, and don't operate on PySpark DataFrames.  Note that for many projects, especially those that leverage PySpark, the best way to interact with Databricks might be with Databricks Connect, not this library.


# Installation


## SQL Connector Only

To include `databricks_warehouse` in your project, add it as a dependency in your project's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
...
databricks_warehouse = { path = "../../libraries/databricks_warehouse", develop = True }
```

## Databricks Connect Fallback

The Databricks SQL Connector can only be used from compute that's running outside of Databricks.  To support queries from Databricks jobs as well, the preferred method is to use Databricks Connect.  All query methods have built-in fallbacks to Databricks Connect which will be run if we detect we're in a Databricks environment.  To include this optional feature, make sure to include the `connect` extra (excluded by default to limit dependencies):

```bash
pip install databricks_warehouse[connect]
```

```bash
poetry add databricks_warehouse --extras connect
```

# Primary Methods

The following methods can be imported from `databricks_warehouse` directly and are the primary ways that users should interact with this library:

1. `read_databricks` -- Run a SQL query against an endpoint and return the results as a Pandas DataFrame.
2. `read_databricks_pl` -- Run a SQL query against an endpoint and return the results as a Polars DataFrame.
3. `execute_databricks` -- Execute a SQL statement and don't return results.  This is useful for non-SELECT statements such as INSERT, CREATE, etc.


# Configuration


## Databricks Client Unified Authentication

This project uses the Databricks SDK for configuring credentials and cluster information.  As such, it adheres to the Databricks Client Unified Authentication protocol: https://docs.databricks.com/en/dev-tools/auth/unified-auth.html.  This means that any setup you already have for configuring Databricks connections (namely environment variables and your `~/.databrickscfg` file) should directly apply here.

Due to security issues with Personal Access Tokens, this library only supports User-to-Machine (U2M) and Machine-to-Machine (M2M) OAuth.


## Environment Variables

Set the following environment variables to configure the connection defaults:

### Authentication

Always Required:
- `DATABRICKS_HOST`: The URL of the Databricks workspace.

Optional:
- `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`: For Machine-to-Machine OAuth using a service principal.  Set this when running from automated jobs as a service principal.

If `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` are not provided, we fall back to User-to-Machine OAuth using the Databricks CLI instead.  See [this documentation](https://docs.databricks.com/en/dev-tools/cli/authentication.html#oauth-user-to-machine-u2m-authentication) for more information.

### Cluster Specification

Either:

- `DATABRICKS_WAREHOUSE_ID`: The ID of the SQL warehouse to run queries on.  This should be the primary way the library is used, as SQL warehouses are highly optimized for queries.  Cannot be set with `DATABRICKS_CLUSTER_ID`.

OR 

- `DATABRICKS_CLUSTER_ID`: The ID of the cluster you wish to connect to.  Cannot be set with `DATABRICKS_WAREHOUSE_ID`.

## Configuration Priority Ordering

As this library leverages the Databricks SDK for configuration, we have the same ability to read settings either from environment variables or from a Databricks Configuration File (`~/.databrickscfg`).  In all cases, environment variables take precedence over configuration profiles.  See [Authentication order of evaluation](https://docs.databricks.com/en/dev-tools/cli/authentication.html#authentication-order-of-evaluation) for more information.


# Examples


```python
from databricks_warehouse import read_databricks, read_databricks_pl, execute_databricks

# By default, the target is determined from environment variables or ~/.databrickscfg settings.
# This will run on whatever $DATABRICKS_CLUSTER_ID / $DATABRICKS_WAREHOUSE_ID specify.
# If running remotely from outside of Databricks, this will use the Databricks SQL Connector.
# If running from a Databricks job, this will use Databricks Spark instead.
df = read_databricks("SELECT * FROM your_table LIMIT 10")

# Settings can also be overridden if you require more flexibility than environment variables allow
df = read_databricks(
    "SELECT * FROM your_table LIMIT 10",
    host="https://your-workspace.cloud.databricks.com",
    warehouse_id="your-warehouse-id",
)

# Same, but returns results as Polars DataFrame instead of Pandas
df_polars = read_databricks_pl("SELECT * FROM your_table LIMIT 10")

# For non-SELECT statements (CREATE, DELETE, INSERT, etc.), use `execute_databricks` instead
execute_databricks("CREATE TABLE dev.my_table (...)")
```
