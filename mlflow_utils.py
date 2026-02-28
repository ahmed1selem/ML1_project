from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


def _to_mlflow_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def setup_experiment(
    tracking_uri: str,
    experiment_name: str,
    artifact_location: Optional[str] = None,
    experiment_tags: Optional[Mapping[str, Any]] = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags={k: _to_mlflow_value(v) for k, v in (experiment_tags or {}).items()},
        )
    else:
        experiment_id = experiment.experiment_id
        for k, v in (experiment_tags or {}).items():
            client.set_experiment_tag(experiment_id, k, _to_mlflow_value(v))

    mlflow.set_experiment(experiment_name)
    return experiment_id


def build_run_name(
    model_name: str,
    seed: int,
    split_name: str = "stratified_80_20",
    feature_set: str = "landmarks63_plus_engineered",
    data_version: str = "v1",
) -> str:
    return f"{model_name}__{feature_set}__{split_name}__seed{seed}__{data_version}"


@contextmanager
def start_run(
    run_name: str,
    tags: Optional[Mapping[str, Any]] = None,
    nested: bool = False,
    description: Optional[str] = None,
):
    with mlflow.start_run(run_name=run_name, nested=nested, description=description) as run:
        if tags:
            mlflow.set_tags({k: _to_mlflow_value(v) for k, v in tags.items()})
        yield run


def log_params(params: Mapping[str, Any], prefix: Optional[str] = None) -> None:
    for key, value in params.items():
        k = f"{prefix}.{key}" if prefix else key
        mlflow.log_param(k, _to_mlflow_value(value))


def log_metrics(
    metrics: Mapping[str, float],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    prepared = {}
    for key, value in metrics.items():
        if value is None:
            continue
        k = f"{prefix}.{key}" if prefix else key
        prepared[k] = float(value)

    if step is None:
        mlflow.log_metrics(prepared)
    else:
        for key, value in prepared.items():
            mlflow.log_metric(key, value, step=step)


def log_model(
    model: Any,
    artifact_path: str = "model",
    X_example: Optional[Any] = None,
    model_flavor: str = "sklearn",
    registered_model_name: Optional[str] = None,
    await_registration_for: int = 300,
) -> str:
    if model_flavor != "sklearn":
        raise ValueError("Only model_flavor='sklearn' is supported.")

    signature = None
    input_example = None

    if X_example is not None:
        input_example = X_example
        try:
            preds = model.predict(input_example)
            signature = infer_signature(input_example, preds)
        except Exception:
            signature = None

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
    )

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active run while logging model.")
    return f"runs:/{active_run.info.run_id}/{artifact_path}"


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def log_dataset_info(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    dataset_path: Optional[str] = None,
    artifact_path: str = "dataset",
    sample_size: int = 1500,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found.")

    info = {
        "dataset_name": dataset_name,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "target_column": target_col,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in df[target_col].value_counts().to_dict().items()},
        "dataset_fingerprint_sha256": _dataset_fingerprint(df),
    }
    if extra_info:
        info.update({k: _to_mlflow_value(v) for k, v in extra_info.items()})

    mlflow.log_dict(info, f"{artifact_path}/dataset_info.json")

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_n = min(sample_size, len(df))
        sample_path = Path(tmp_dir) / f"{dataset_name}_sample.csv"
        df.sample(n=sample_n, random_state=42).to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path=artifact_path)

    if dataset_path and Path(dataset_path).exists():
        mlflow.log_artifact(dataset_path, artifact_path=artifact_path)

    return info


def log_artifacts(paths: Iterable[str], artifact_path: Optional[str] = None) -> None:
    for p in paths:
        path = Path(p)
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        elif path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            raise FileNotFoundError(f"Artifact path does not exist: {path}")


def log_model_comparison_plot(
    results_df: pd.DataFrame,
    metric_cols: list[str],
    model_col: str = "model_name",
    sort_by: Optional[str] = "f1_macro",
    artifact_path: str = "comparison",
    filename: str = "model_comparison.png",
    title: str = "Model Comparison",
) -> str:
    needed = [model_col] + metric_cols
    missing = [c for c in needed if c not in results_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    plot_df = results_df[needed].copy()
    if sort_by and sort_by in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.8 / len(metric_cols)

    fig_w = max(10, len(plot_df) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    for i, metric in enumerate(metric_cols):
        offset = -0.4 + (i + 0.5) * width
        ax.bar(x + offset, plot_df[metric].values, width=width, label=metric)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[model_col].tolist(), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = Path(tmp_dir) / filename
        csv_path = Path(tmp_dir) / "model_comparison.csv"
        fig.savefig(img_path, dpi=160, bbox_inches="tight")
        plot_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(img_path), artifact_path=artifact_path)
        mlflow.log_artifact(str(csv_path), artifact_path=artifact_path)

    plt.close(fig)
    return f"{artifact_path}/{filename}"


def select_best_model(
    results_df: pd.DataFrame,
    primary_metric: str = "f1_macro",
    higher_is_better: bool = True,
    tie_breaker_metric: Optional[str] = "accuracy",
    tie_breaker_higher_is_better: bool = True,
) -> pd.Series:
    if primary_metric not in results_df.columns:
        raise ValueError(f"primary_metric '{primary_metric}' missing from results_df")

    sort_cols = [primary_metric]
    ascending = [not higher_is_better]
    if tie_breaker_metric and tie_breaker_metric in results_df.columns and tie_breaker_metric != primary_metric:
        sort_cols.append(tie_breaker_metric)
        ascending.append(not tie_breaker_higher_is_better)

    ranked = results_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return ranked.iloc[0]


def register_model(
    model_uri: str,
    model_name: str,
    model_version_tags: Optional[Mapping[str, Any]] = None,
    model_description: Optional[str] = None,
    alias: Optional[str] = None,
    await_registration_for: int = 300,
) -> dict:
    client = MlflowClient()

    registration = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        await_registration_for=await_registration_for,
    )

    if model_description:
        client.update_registered_model(name=model_name, description=model_description)

    for key, value in (model_version_tags or {}).items():
        client.set_model_version_tag(
            name=model_name,
            version=registration.version,
            key=key,
            value=_to_mlflow_value(value),
        )

    if alias:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=registration.version,
        )

    return {
        "name": registration.name,
        "version": int(registration.version),
        "model_uri": f"models:/{registration.name}/{registration.version}",
    }
