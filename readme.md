# MLflow UI Visibility Audit Report
Date: 2026-02-28
Project: E:\ITI\ML1\fina_project
Scope: UI visibility of experiments, tags, run logs, and registered models. No retraining. No source-code changes.

## Executive Summary
- The active MLflow UI on port `5000` is serving the `fina_project` store: `file:///E:/ITI/ML1/fina_project/mlruns`.
- In this store, tags and model logs are present, but not all are visible in list views by default.
- "Other logs/models" are in a different tracking store: `file:///E:/ITI/ML1/project/mlruns`.
- This is a data-store separation issue, not a training or logging failure.

## Evidence (Screenshots)
1. Experiments in active store (`fina_project`):
   - ![Experiments](./report/mlflow_experiments.png)
2. Baselines experiment run list (7 runs):
   - ![Baselines Runs](./report/mlflow_runs_baselines.png)
3. Run detail page showing tags + registered model link:
   - ![Run Detail Tags](./report/mlflow_run_detail_tags.png)
4. Registered Models list in active store:
   - ![Models](./report/mlflow_models.png)
5. Model version page showing version tags:
   - ![Model Version Tags](./report/mlflow_model_version1.png)
6. Experiments in older `project` store (contains the other logs):
   - ![Old Store Experiments](./report/mlflow_old_experiments.png)
7. Runs in older `project_run` experiment:
   - ![Old Store Runs](./report/mlflow_old_runs.png)

## What Is Present in `fina_project` Store
- Experiment: `hand_gesture/hagrid_landmarks/baselines`
- Run count: 7
- Run-level tags exist (examples):
  - `stage=train_eval`
  - `dataset_version=hagrid_landmarks_v1`
  - `model_name=XGBoost`
- Registered model: `hand_gesture_hagrid_landmarks_classifier`
- Model version: `1` (alias: `champion`)
- Version tags exist (examples):
  - `selected_model_name=XGBoost`
  - `selection_metric=f1_macro`
  - `dataset_version=hagrid_landmarks_v1`

## Why Tags / Other Logs Were Not Obvious in UI
- MLflow list pages do not show every tag by default.
- Model-level `tags` are empty in this run set; useful metadata was logged at **model version tags** level.
- Older experiments (`project_run`, `smoke_test`) were logged to a separate store (`E:/ITI/ML1/project/mlruns`), so they do not appear in the `fina_project` UI instance.

## Non-Destructive Recommendations
1. Keep using explicit backend URI when launching UI:
   - `mlflow ui --backend-store-uri "file:///E:/ITI/ML1/fina_project/mlruns" --port 5000`
2. In run table, add columns for:
   - `tags.stage`, `tags.dataset_version`, `tags.model_name`
3. In model pages, inspect **version tags** (not only top-level model tags).
4. If you want a single pane of glass, standardize on one store path moving forward.

## Deliverables
- Report file: `reports/mlflow_ui_audit_2026-02-28/MLFLOW_UI_AUDIT_REPORT.md`
- Screenshot bundle: `reports/mlflow_ui_audit_2026-02-28/*.png`
