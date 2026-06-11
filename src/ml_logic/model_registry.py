"""
Model persistence and registry.

Handles saving/loading trained models and their metadata (best hyperparameters,
CV scores) to/from the artifacts/ directory.
"""

import joblib
import yaml
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def save_model(
    model,
    name: str,
    params: dict | None = None,
    cv_score: float | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """Save a fitted model and its metadata.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible estimator.
    name : str
        Model identifier (e.g., "model_A"). Creates artifacts/<name>/.
    params : dict, optional
        Best hyperparameters found during search.
    cv_score : float, optional
        Best cross-validation score.
    extra_metadata : dict, optional
        Any additional metadata to persist.

    Returns
    -------
    Path
        Directory where the model was saved.
    """
    model_dir = ARTIFACTS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the fitted estimator
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save metadata
    metadata = {}
    if params is not None:
        # Convert numpy types to native Python for YAML serialization
        metadata["best_params"] = {
            k: _to_native(v) for k, v in params.items()
        }
    if cv_score is not None:
        metadata["cv_score"] = float(cv_score)
    if extra_metadata:
        metadata.update(extra_metadata)

    if metadata:
        params_path = model_dir / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"Params saved to {params_path}")

    return model_dir


def load_model(name: str):
    """Load a single trained model by name.

    Parameters
    ----------
    name : str
        Model identifier (e.g., "model_A").

    Returns
    -------
    estimator
        The fitted model.
    """
    model_path = ARTIFACTS_DIR / name / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            f"Train and save the model first."
        )
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model


def load_model_metadata(name: str) -> dict:
    """Load the metadata (params, CV score) for a model.

    Returns
    -------
    dict
        Metadata dictionary. Empty dict if no params.yaml exists.
    """
    params_path = ARTIFACTS_DIR / name / "params.yaml"
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        return yaml.safe_load(f) or {}


def load_all_models() -> dict:
    """Load all trained models from the artifacts directory.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name -> fitted estimator.
    """
    models = {}
    for model_dir in sorted(ARTIFACTS_DIR.iterdir()):
        model_path = model_dir / "model.joblib"
        if model_dir.is_dir() and model_path.exists():
            models[model_dir.name] = joblib.load(model_path)
            print(f"Loaded: {model_dir.name}")

    if not models:
        print("Warning: No trained models found in artifacts/.")
    else:
        print(f"Total models loaded: {len(models)}")

    return models


def _to_native(val):
    """Convert numpy/other types to native Python for YAML serialization."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
