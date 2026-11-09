

# COMMAND ----------
# 1. Imports
import os
import warnings
from typing import Optional, Dict, Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from mlflow.models import infer_signature

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("default")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    print("XGBoost não está instalado; o modelo será omitido.")

# COMMAND ----------
import yaml

env = "dev"

with open('../config.yml') as f:
    config_dict = yaml.safe_load(f)
    CATALOG = config_dict[env]["catalog_name"]
    SCHEMA = config_dict[env]["schema_name"]

OUTPUT_SCHEMA = f"{CATALOG}.{SCHEMA}"


TARGET_COLUMN = "big_falha_24h"


train_table = f"{OUTPUT_SCHEMA}.train_set"
test_table = f"{OUTPUT_SCHEMA}.test_set"

# Carregar tabelas
df_train = spark.table(train_table)
df_test = spark.table(test_table)

# Converter para Pandas
df_train = df_train.toPandas()
df_test = df_test.toPandas()

# Separar features e target
X_train = df_train.drop(columns=[TARGET_COLUMN])
y_train = df_train[TARGET_COLUMN]

X_test = df_test.drop(columns=[TARGET_COLUMN])
y_test = df_test[TARGET_COLUMN]

# COMMAND ----------


TARGET_COLUMN = "big_falha_24h"

EXPERIMENT_NAME = "/Shared/DataPulse/FailurePrediction"

# Para registrar versões no Unity Catalog, defina por exemplo:
# REGISTERED_MODEL_NAME = "datapulse.ml.failure_predictor"
# O registro é opcional e requer permissões no Unity Catalog.
REGISTERED_MODEL_NAME = None

mlflow.set_experiment(EXPERIMENT_NAME)
print("MLflow experiment:", EXPERIMENT_NAME)
print("Tracking URI:", mlflow.get_tracking_uri())

# Teste de hiperparametros

PARAM_GRID = {
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
    },

    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },

    "SVC": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },

    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    },

    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },

    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [5000, 10000],
    },

    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
}

if XGBOOST_AVAILABLE:
    PARAM_GRID["XGBClassifier"] = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


# COMMAND ----------
# 5. Registry centralizado dos estimadores
MODEL_REGISTRY = {
    "GaussianNB": GaussianNB,
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
}

if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["XGBClassifier"] = XGBClassifier

print("Modelos disponíveis:", list(MODEL_REGISTRY.keys()))

# COMMAND ----------

# 6. Treinamento, otimização, avaliação e logging no MLflow

def train_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    registered_model_name: Optional[str] = REGISTERED_MODEL_NAME,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
) -> Dict[str, Any]:

    """
    Otimiza hiperparâmetros com GridSearchCV, treina o melhor
    estimador e registra parâmetros, métricas e artefatos no MLflow.
    """

    with mlflow.start_run(run_name=model_name) as run:

        mlflow.set_tags({
            "project": "DataPulse",
            "task": "industrial_failure_classification",
            "model_name": model_name,
            "target_column": TARGET_COLUMN,
            "optimization": "GridSearchCV",
        })

        # --------------------------------------------------
        # 1. Configuração da validação cruzada
        # --------------------------------------------------

        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        if param_grid is None:
            param_grid = PARAM_GRID.get(type(model).__name__, {})

        # --------------------------------------------------
        # 2. Otimização dos hiperparâmetros
        # --------------------------------------------------

        print(f"\nStarting GridSearchCV: {model_name}")

        if param_grid:

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv_strategy,
                n_jobs=n_jobs,
                verbose=1,
                refit=True,
                error_score="raise",
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = float(grid_search.best_score_)

            print("Best parameters:", best_params)
            print("Best CV score:", best_score)

            # Registrar hiperparâmetros encontrados
            mlflow.log_params({
                f"best_{key}": value
                for key, value in best_params.items()
            })

            mlflow.log_metric("best_cv_score", best_score)

            # Salvar resultados completos da busca
            cv_results = pd.DataFrame(grid_search.cv_results_)

            cv_results_path = "/tmp/grid_search_results.csv"
            cv_results.to_csv(cv_results_path, index=False)

            mlflow.log_artifact(
                cv_results_path,
                artifact_path="optimization",
            )

        else:
            print("No parameter grid found. Using default parameters.")

            best_model = model
            best_model.fit(X_train, y_train)

            best_params = best_model.get_params()
            best_score = None

        # --------------------------------------------------
        # 3. Registrar parâmetros do modelo
        # --------------------------------------------------

        mlflow.log_params({
            f"final_{key}": str(value)
            for key, value in best_model.get_params().items()
            if value is not None
        })

        mlflow.log_params({
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": X_train.shape[1],
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": cv,
            "optimization_scoring": scoring,
        })

        # --------------------------------------------------
        # 4. Avaliação no conjunto de teste
        # --------------------------------------------------

        y_pred = best_model.predict(X_test)

        average = (
            "binary"
            if y_test.nunique() == 2
            else "weighted"
        )

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    average=average,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    average=average,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    average=average,
                    zero_division=0,
                )
            ),
        }

        # ROC AUC
        try:

            if y_test.nunique() == 2:

                if hasattr(best_model, "predict_proba"):

                    proba = best_model.predict_proba(X_test)

                    positive_label = (
                        1 if 1 in list(best_model.classes_)
                        else best_model.classes_[-1]
                    )

                    positive_idx = list(
                        best_model.classes_
                    ).index(positive_label)

                    y_binary = (
                        y_test == positive_label
                    ).astype(int)

                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_binary,
                            proba[:, positive_idx],
                        )
                    )

                elif hasattr(best_model, "decision_function"):

                    scores = best_model.decision_function(X_test)

                    metrics["roc_auc"] = float(
                        roc_auc_score(y_test, scores)
                    )

        except Exception as exc:
            print(f"Warning: ROC AUC unavailable: {exc}")

        mlflow.log_metrics(metrics)

        # --------------------------------------------------
        # 5. Registrar modelo no MLflow
        # --------------------------------------------------

        signature = infer_signature(X_test, y_pred)

        input_example = X_test.head(
            min(5, len(X_test))
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        # --------------------------------------------------
        # 6. Relatório de classificação
        # --------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        report_path = "/tmp/classification_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        mlflow.log_artifact(
            report_path,
            artifact_path="evaluation",
        )

        # --------------------------------------------------
        # 7. Matriz de confusão
        # --------------------------------------------------

        labels = sorted(
            pd.Series(y_test).unique().tolist()
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
        )

        cm_path = "/tmp/confusion_matrix.csv"

        cm_df.to_csv(cm_path, index=True)

        mlflow.log_artifact(
            cm_path,
            artifact_path="evaluation",
        )

        # --------------------------------------------------
        # 8. Model Registry
        # --------------------------------------------------

        registry_version = None

        if registered_model_name:

            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_model_name,
            )

            registry_version = str(registered.version)

        # --------------------------------------------------
        # 9. Resultado
        # --------------------------------------------------

        result = {
            "model_name": model_name,
            "model_type": type(best_model).__name__,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "registered_model_version": registry_version,
            "best_params": best_params,
            "best_cv_score": best_score,
            **metrics,
        }

        print(f"Run ID: {run.info.run_id}")
        print("Best parameters:", best_params)
        print("Metrics:", metrics)
        print("Classification report:\n", report)

        return result


# COMMAND ----------
# 7. Executar um ou vários modelos a partir de configurações
def run_experiment(
    models: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """Executa os modelos declarados no dicionário de configuração."""
    

    if models is None:
        models = {
            "GaussianNB_v1": {
                "model": "GaussianNB",
                "parameters": {},
            }
        }

    results = []
    for model_name, model_config in models.items():
        model_type = model_config["model"]
        parameters = model_config.get("parameters", {})

        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Modelo '{model_type}' não está disponível. "
                f"Modelos disponíveis: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[model_type]
        model = model_class(**parameters)

        print(f"\n{'=' * 60}")
        print(f"Training model: {model_name}")
        print(f"Model type: {model_type}")
        print(f"Parameters: {parameters}")
        print(f"{'=' * 60}")

        result = train_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty and "f1" in results_df.columns:
        results_df = results_df.sort_values("f1", ascending=False).reset_index(drop=True)

    return results_df


# COMMAND ----------
# 8. Executar todos os experimentos
def run_allexperiments() -> Dict[str, Any]:
    """Executa a lista padrão de modelos usando o mesmo split e o mesmo pipeline MLflow."""
    print("Starting model training...")

    models = {
        "GaussianNB_v3": {
            "model": "GaussianNB",
            "parameters": {},
        },
        "KNN_v3": {
            "model": "KNeighborsClassifier",
            "parameters": {"n_neighbors": 5, "weights": "distance"},
        },
        "SVM_v3": {
            "model": "SVC",
            "parameters": {
                "C": 1.0, "kernel": "rbf",
                "probability": True, "random_state": 42,
            },
        },
        "GradientBoosting_v3": {
            "model": "GradientBoostingClassifier",
            "parameters": {
                "n_estimators": 200, "learning_rate": 0.05,
                "max_depth": 5, "random_state": 42,
            },
        },
        "RandomForest_v3": {
            "model": "RandomForestClassifier",
            "parameters": {
                "n_estimators": 300, "max_depth": 10,
                "min_samples_split": 10, "random_state": 42,
                "n_jobs": -1,
            },
        },
        "LogisticRegression_v3": {
            "model": "LogisticRegression",
            "parameters": {
                "max_iter": 10000, "C": 1.0, "random_state": 42,
            },
        },
        "DecisionTree_v3": {
            "model": "DecisionTreeClassifier",
            "parameters": {
                "max_depth": 5, "min_samples_split": 10,
                "random_state": 42,
            },
        },
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost_v3"] = {
            "model": "XGBClassifier",
            "parameters": {
                "n_estimators": 300, "learning_rate": 0.1,
                "max_depth": 6, "random_state": 42,
                "n_jobs": -1, "eval_metric": "logloss",
            },
        }

    results_df = run_experiment(
        models=models
    )

    print("All experiments executed successfully.")
    display(results_df)

    return {
        "status": "success",
        "total_experiments": len(results_df),
        "results": results_df,
    }



# COMMAND ----------
all_results = run_allexperiments()
print(all_results["status"])
print("Total de experimentos:", all_results["total_experiments"])

# COMMAND ----------
def selecionar_melhor_run(
    experiment_name: str,
    metricas: list = None,
    ascending: bool = False
) -> pd.DataFrame:

    """
    Consulta todas as Runs de um experimento MLflow
    e classifica os modelos pelas métricas selecionadas.

    Critério padrão: média das métricas.
    """

    if metricas is None:
        metricas = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise ValueError(
            f"Experimento não encontrado: {experiment_name}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=10000
    )

    resultados = []

    for run in runs:

        metrics = run.data.metrics

        # Verifica se todas as métricas existem
        if not all(
            metric in metrics
            for metric in metricas
        ):
            continue

        resultado = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get(
                "mlflow.runName", "N/A"
            ),
            "model": run.data.params.get(
                "model", "N/A"
            )
        }

        for metric in metricas:
            resultado[metric] = metrics[metric]

        # Média das métricas
        resultado["score"] = sum(
            metrics[metric]
            for metric in metricas
        ) / len(metricas)

        resultados.append(resultado)

    if not resultados:
        raise ValueError(
            "Nenhuma Run possui todas as métricas solicitadas."
        )

    df_resultados = pd.DataFrame(resultados)

    df_resultados = df_resultados.sort_values(
        by="score",
        ascending=ascending
    ).reset_index(drop=True)

    df_resultados.index += 1
    df_resultados.index.name = "ranking"

    return df_resultados

# COMMAND ----------
def validar_modelo(metrics):

    if metrics["f1"] < 0.85:
        raise ValueError(
            "Modelo reprovado: F1 abaixo do mínimo"
        )

    if metrics["recall"] < 0.90:
        raise ValueError(
            "Modelo reprovado: Recall abaixo do mínimo"
        )

    if metrics["accuracy"] < 0.90:
        raise ValueError(
            "Modelo reprovado: Accuracy abaixo do mínimo"
        )

    if metrics["precision"] < 0.90:
        raise ValueError(
            "Modelo reprovado: precision abaixo do mínimo"
        )

    print("Modelo aprovado!")

    return True


# COMMAND ----------
df_ranking = selecionar_melhor_run(
    experiment_name=EXPERIMENT_NAME
)

print(df_ranking)

# COMMAND ----------
# Buscar os detalhes da execução (run), que incluem as métricas

run_id = str(df_ranking["run_id"].iloc[0])
model_uri = f"runs:/{run_id}/model"
# Inicializar o cliente do MLflow
client = MlflowClient()
run = client.get_run(run_id)
metrics = run.data.metrics
print("Métricas:", metrics)
# Validar o modelo
validar_modelo(metrics)

# COMMAND ----------
#LogisticRegression_v2
model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="datapulse.ml.failure_predictor_DecisionTree_v1",
)

print("Versão registrada:", registered_model.version)

# COMMAND ----------

# COMMAND ----------