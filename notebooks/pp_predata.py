# Databricks notebook source
import sys
# import pyspark
# import mlflow

print(sys.version)
print(sys.executable)
print("Ambiente MLOps configurado!")
# COMMAND ----------
%pip install -e .. 
%restart_python

# COMMAND ----------
# Opção recomendada no Databricks: tabela Gold do Unity Catalog.
DATA_SOURCE = "datapulse.gold.sensor_events"
# 2. Configuração do notebook


# Alternativa CSV: informe um caminho acessível ao driver do cluster,
# por exemplo "/Volumes/main/default/industrial_data/industrial_data_serving.csv".
# Não use um caminho local do seu computador Windows.
DATA_SOURCE_TYPE = "table"  # "table" ou "csv"


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
# 4. Carregamento, divisão e persistência dos conjuntos de treino e teste
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------
# 3. Carregamento e validação inicial dos dados
def load_data(source: str = DATA_SOURCE, source_type: str = DATA_SOURCE_TYPE) -> pd.DataFrame:
    """Carrega dados de uma tabela Spark/Unity Catalog ou de um CSV."""
    if source_type == "table":
        # Disponível dentro de um notebook Databricks com Spark ativo.
        sdf = spark.table(source)
        df = sdf.toPandas()
    elif source_type == "csv":
        df = pd.read_csv(source)
    else:
        raise ValueError("source_type deve ser 'table' ou 'csv'.")

    if df.empty:
        raise ValueError(f"A fonte de dados '{source}' está vazia.")

    print(f"Fonte: {source}")
    print(f"Shape: {df.shape}")
    print("Schema:")
    print(df.dtypes)
    return df


# COMMAND ----------

def data_split(
    df,
    target="big_falha_24h",
    test_size=0.20,
    random_state=42
):
    df = df.copy()

    if target not in df.columns:
        raise ValueError(
            f"A coluna alvo '{target}' não existe no DataFrame."
        )

    # Remove registros sem a variável alvo
    df = df.dropna(subset=[target])

    # Separa features e alvo
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Converte colunas com nomes relacionados a datas
    date_columns = [
        col for col in X.columns
        if any(term in col.lower() for term in
               ["data", "date", "timestamp", "datetime"])
    ]

    for col in date_columns:
        parsed = pd.to_datetime(X[col], errors="coerce")

        # Só transforma a coluna se ela realmente representar datas
        if parsed.notna().mean() >= 0.5:
            X[f"{col}_ano"] = parsed.dt.year
            X[f"{col}_mes"] = parsed.dt.month
            X[f"{col}_dia"] = parsed.dt.day
            X[f"{col}_dia_semana"] = parsed.dt.dayofweek

            # Remove a data original em formato textual
            X = X.drop(columns=[col])

    # Converte booleanos em 0 e 1
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # Trata infinitos
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove colunas totalmente vazias
    X = X.dropna(axis=1, how="all")

    # Identifica colunas categóricas
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # Divide antes de aprender imputações e codificações
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Codifica categorias sem ajustar usando o conjunto de teste
    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train[categorical_cols] = encoder.fit_transform(
            X_train[categorical_cols].fillna("__AUSENTE__").astype(str)
        )

        X_test[categorical_cols] = encoder.transform(
            X_test[categorical_cols].fillna("__AUSENTE__").astype(str)
        )

    # Imputação numérica ajustada somente no treino
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train, X_test, y_train, y_test


# COMMAND ----------
import yaml

env = "test"

with open('../config.yml') as f:
    config_dict = yaml.safe_load(f)
    CATALOG = config_dict[env]["catalog_name"]
    SCHEMA = config_dict[env]["schema_name"]

OUTPUT_SCHEMA = f"{CATALOG}.{SCHEMA}"


TARGET_COLUMN = "big_falha_24h"

df = load_data(source = DATA_SOURCE, source_type = DATA_SOURCE_TYPE)
X_train, X_test, y_train, y_test = data_split(df, target=TARGET_COLUMN)

train_set = pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1)
test_set = pd.concat([X_test, y_test.rename(TARGET_COLUMN)], axis=1)

processing_timestamp = datetime.now()
train_sdf = spark.createDataFrame(train_set).withColumn(
    "datahora_processamento", F.lit(processing_timestamp)
)
test_sdf = spark.createDataFrame(test_set).withColumn(
    "datahora_processamento", F.lit(processing_timestamp)
)

train_table = f"{OUTPUT_SCHEMA}.train_set"
test_table = f"{OUTPUT_SCHEMA}.test_set"

train_sdf.write.format("delta").mode("overwrite").saveAsTable(train_table)
test_sdf.write.format("delta").mode("overwrite").saveAsTable(test_table)

print(f"Tabela de treino salva: {train_table}")
print(f"Tabela de teste salva: {test_table}")

