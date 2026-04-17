"""
Optuna Classification Tuning Pipeline
====================================
Tunes preprocessing (PCA, scaling) and model hyperparameters
for a classification task using Optuna's Bayesian optimisation.

Usage
-----
    from optuna_classification_tuner import ClassificationTuner

    tuner = ClassificationTuner(n_trials=100, cv=5, scoring="f1_macro")
    tuner.fit(X_train, y_train)
    tuner.print_results()

    best_pipeline = tuner.best_pipeline_
    predictions   = best_pipeline.predict(X_test)

Type-checking
-------------
    mypy  optuna_classification_tuner.py --strict
    pyright optuna_classification_tuner.py
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from argparse import ArgumentParser
import pandas as pd

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Domain type aliases
# ---------------------------------------------------------------------------

#: Any sklearn-compatible classifier.
Classifier: TypeAlias = ClassifierMixin

#: Any sklearn-compatible transformer (scaler, PCA, …).
Transformer: TypeAlias = TransformerMixin

#: A single named pipeline step, e.g. ``("scaler", StandardScaler())``.
PipelineStep: TypeAlias = tuple[str, BaseEstimator]

#: The raw Optuna params dict returned by ``trial.params``.
ParamsDict: TypeAlias = dict[str, Any]

#: Supported scaler choices surfaced to Optuna.
ScalerChoice: TypeAlias = Literal["standard", "maxabs", "none"]

#: Supported model names surfaced to Optuna.
ModelName: TypeAlias = Literal[
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svc",
    "knn",
]

#: Optimisation direction.
Direction: TypeAlias = Literal["maximize", "minimize"]

#: Float array shaped ``(n_samples, n_features)``.
FeatureMatrix: TypeAlias = NDArray[np.float64]

#: Float array shaped ``(n_samples,)``.
TargetVector: TypeAlias = NDArray[np.float64]


# ---------------------------------------------------------------------------
# TypedDicts for structured return values
# ---------------------------------------------------------------------------


class TrialSummary(TypedDict):
    """One entry in the list returned by :meth:`ClassificationTuner.top_trials`."""

    rank: int
    score: float
    params: ParamsDict


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------


def _build_model(trial: optuna.Trial) -> tuple[ModelName, Classifier]:
    """
    Suggest a classification model and its hyperparameters.

    Parameters
    ----------
    trial:
        Active Optuna trial.

    Returns
    -------
    tuple[ModelName, Classifier]
        ``(model_name, unfitted_estimator)``
    """
    model_name: ModelName = trial.suggest_categorical(  # type: ignore[assignment]
        "model",
        [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "svc",
            "knn",
        ],
    )

    model: Classifier

    if model_name == "logistic_regression":
        penalty: Literal["l2", "none"] = trial.suggest_categorical(
            "logistic_regression__penalty", ["l2", "none"]
        )
        C: float = trial.suggest_float("logistic_regression__C", 1e-3, 1e3, log=True)
        solver: Literal["lbfgs", "saga"] = trial.suggest_categorical(
            "logistic_regression__solver", ["lbfgs", "saga"]
        )
        model = LogisticRegression(
            penalty=None if penalty == "none" else penalty,
            C=C,
            solver=solver,
            max_iter=5_000,
            random_state=42,
        )

    elif model_name == "decision_tree":
        max_depth: int = trial.suggest_int("dt__max_depth", 2, 20)
        min_samples_split: int = trial.suggest_int("dt__min_samples_split", 2, 20)
        min_samples_leaf: int = trial.suggest_int("dt__min_samples_leaf", 1, 10)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

    elif model_name == "random_forest":
        n_estimators: int = trial.suggest_int("rf__n_estimators", 50, 500)
        max_depth = trial.suggest_int("rf__max_depth", 2, 20)
        min_samples_split = trial.suggest_int("rf__min_samples_split", 2, 20)
        max_features: float = trial.suggest_float("rf__max_features", 0.3, 1.0)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "gradient_boosting":
        n_estimators = trial.suggest_int("gb__n_estimators", 50, 500)
        max_depth = trial.suggest_int("gb__max_depth", 2, 8)
        learning_rate: float = trial.suggest_float(
            "gb__learning_rate", 0.01, 0.3, log=True
        )
        subsample: float = trial.suggest_float("gb__subsample", 0.5, 1.0)
        min_samples_split = trial.suggest_int("gb__min_samples_split", 2, 20)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            random_state=42,
        )

    elif model_name == "svc":
        C: float = trial.suggest_float("svc__C", 1e-2, 1e3, log=True)
        kernel: Literal["linear", "rbf", "poly"] = trial.suggest_categorical(  # type: ignore[assignment]
            "svc__kernel", ["linear", "rbf", "poly"]
        )
        svc_kwargs: dict[str, Any] = {"C": C, "kernel": kernel}
        if kernel == "rbf":
            svc_kwargs["gamma"] = trial.suggest_float("svc__gamma", 1e-4, 1.0, log=True)
        elif kernel == "poly":
            svc_kwargs["degree"] = trial.suggest_int("svc__degree", 2, 4)
        model = SVC(**svc_kwargs)

    elif model_name == "knn":
        n_neighbors: int = trial.suggest_int("knn__n_neighbors", 1, 30)
        weights: Literal["uniform", "distance"] = trial.suggest_categorical(  # type: ignore[assignment]
            "knn__weights", ["uniform", "distance"]
        )
        p: int = trial.suggest_int("knn__p", 1, 2)  # 1 = Manhattan, 2 = Euclidean
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    return model_name, model


# ---------------------------------------------------------------------------
# Preprocessing catalogue
# ---------------------------------------------------------------------------


def _build_preprocessor(trial: optuna.Trial, n_features: int) -> list[PipelineStep]:
    """
    Suggest scaling and optional PCA steps.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    n_features:
        Number of input features (upper bound for PCA components).

    Returns
    -------
    list[PipelineStep]
        Named ``(step_name, transformer)`` pairs ready for ``sklearn.Pipeline``.
    """
    steps: list[PipelineStep] = []

    # ── Scaling ──────────────────────────────────────────────────────────────
    scaler_choice: ScalerChoice = trial.suggest_categorical(  # type: ignore[assignment]
        "scaler", ["standard", "maxabs", "none"]
    )
    scaler: Transformer | None = None

    if scaler_choice == "standard":
        scaler = StandardScaler()
    elif scaler_choice == "maxabs":
        scaler = MaxAbsScaler()

    if scaler is not None:
        steps.append(("scaler", scaler))

    # ── PCA ──────────────────────────────────────────────────────────────────
    use_pca: bool = trial.suggest_categorical("use_pca", [True, False])  # type: ignore[assignment]
    if use_pca:
        max_components: int = max(1, 20)
        n_components: int = trial.suggest_int("pca__n_components", 1, max_components)
        steps.append(("pca", PCA(n_components=n_components)))

    return steps


# ---------------------------------------------------------------------------
# Main tuner class
# ---------------------------------------------------------------------------


class ClassificationTuner:
    """
    Optuna-powered classification tuner.

    Parameters
    ----------
    n_trials:
        Number of Optuna trials (default ``100``).
    cv:
        Number of cross-validation folds (default ``5``).
    scoring:
        sklearn scoring string, e.g. ``"f1_macro"``, ``"f1_weighted"``
        ``"neg_mean_absolute_error"``, ``"r2"``
        (default ``"neg_root_mean_squared_error"``).
    direction:
        ``"maximize"`` or ``"minimize"``.  Inferred automatically from
        *scoring* when ``None``.
    study_name:
        Optional Optuna study name.
    random_state:
        Seed for reproducibility (default ``42``).
    verbose:
        Print a results summary after optimisation (default ``True``).

    Attributes
    ----------
    study_ : optuna.Study
        The completed Optuna study (set after :meth:`fit`).
    best_pipeline_ : Pipeline
        Best sklearn pipeline refitted on the full training data
        (set after :meth:`fit`).
    best_params_ : ParamsDict
        Hyperparameters of the best trial (set after :meth:`fit`).
    best_score_ : float
        CV score of the best trial (set after :meth:`fit`).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_trials: int = 100,
        cv: int = 5,
        scoring: str = "f1_macro",
        direction: Direction | None = None,
        study_name: str | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.n_trials: int = n_trials
        self.cv: int = cv
        self.scoring: str = scoring
        self.random_state: int = random_state
        self.verbose: bool = verbose
        self.study_name: str = study_name or "classification_tuning"

        # Auto-infer direction
        self.direction: Direction
        if direction is not None:
            self.direction = direction
        else:
            self.direction = "maximize"

        # Populated by fit()
        self.study_: optuna.Study | None = None
        self.best_pipeline_: Pipeline | None = None
        self.best_params_: ParamsDict | None = None
        self.best_score_: float | None = None
        self._X: FeatureMatrix | None = None
        self._y: TargetVector | None = None

    # ------------------------------------------------------------------
    # Internal objective
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        assert (
            self._X is not None and self._y is not None
        ), "_X and _y must be set before calling _objective"

        n_features: int = self._X.shape[1]
        pre_steps: list[PipelineStep] = _build_preprocessor(trial, n_features)
        _name, model = _build_model(trial)

        pipeline: Pipeline = Pipeline(pre_steps + [("model", model)])
        scores: NDArray[np.float64] = cross_val_score(
            pipeline,
            self._X,
            self._y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ClassificationTuner":
        """
        Run the Optuna study and refit the best pipeline on all data.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.
        y:
            Target vector of shape ``(n_samples,)``.

        Returns
        -------
        ClassificationTuner
            ``self`` (for method chaining).
        """
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)

        sampler: optuna.samplers.TPESampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )
        self.study_ = optuna.create_study(
            storage="sqlite:///tuning/classification_tuning.db",
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            load_if_exists=True,
        )
        self.study_.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best: optuna.trial.FrozenTrial = self.study_.best_trial
        self.best_params_ = best.params
        self.best_score_ = float(best.value)  # type: ignore[arg-type]

        # Reconstruct the best pipeline and refit on full data
        n_features: int = self._X.shape[1]
        pre_steps: list[PipelineStep] = self._build_from_params(
            self.best_params_, n_features
        )
        _name, best_model = self._model_from_params(self.best_params_)
        self.best_pipeline_ = Pipeline(pre_steps + [("model", best_model)])
        self.best_pipeline_.fit(self._X, self._y)

        if self.verbose:
            self.print_results()

        return self

    # ------------------------------------------------------------------
    # Pipeline reconstruction from stored params
    # ------------------------------------------------------------------

    @staticmethod
    def _build_from_params(params: ParamsDict, n_features: int) -> list[PipelineStep]:
        """Reconstruct preprocessing steps from a completed trial's params dict."""
        steps: list[PipelineStep] = []

        scaler_choice: ScalerChoice = params.get("scaler", "none")
        if scaler_choice == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaler_choice == "maxabs":
            steps.append(("scaler", MaxAbsScaler()))

        use_pca: bool = bool(params.get("use_pca", False))
        if use_pca:
            n_comp: int = int(params.get("pca__n_components", min(5, n_features - 1)))
            steps.append(("pca", PCA(n_components=n_comp)))

        return steps

    @staticmethod
    def _model_from_params(params: ParamsDict) -> tuple[ModelName, Classifier]:
        """Reconstruct an unfitted classifier from a completed trial's params dict."""
        mn: ModelName = params["model"]

        if mn == "logistic_regression":
            return mn, LogisticRegression(
                penalty=params["logistic_regression__penalty"],
                C=float(params["logistic_regression__C"]),
                solver=params["logistic_regression__solver"],
                max_iter=5_000,
                random_state=42,
            )

        if mn == "decision_tree":
            return mn, DecisionTreeClassifier(
                max_depth=int(params["dt__max_depth"]),
                min_samples_split=int(params["dt__min_samples_split"]),
                min_samples_leaf=int(params["dt__min_samples_leaf"]),
                random_state=42,
            )

        if mn == "random_forest":
            return mn, RandomForestClassifier(
                n_estimators=int(params["rf__n_estimators"]),
                max_depth=int(params["rf__max_depth"]),
                min_samples_split=int(params["rf__min_samples_split"]),
                max_features=float(params["rf__max_features"]),
                random_state=42,
                n_jobs=-1,
            )

        if mn == "gradient_boosting":
            return mn, GradientBoostingClassifier(
                n_estimators=int(params["gb__n_estimators"]),
                max_depth=int(params["gb__max_depth"]),
                learning_rate=float(params["gb__learning_rate"]),
                subsample=float(params["gb__subsample"]),
                min_samples_split=int(params["gb__min_samples_split"]),
                random_state=42,
            )

        if mn == "svc":
            kernel: Literal["linear", "rbf", "poly"] = params["svc__kernel"]
            svc_kwargs: dict[str, Any] = {
                "C": float(params["svc__C"]),
                "kernel": kernel,
            }
            if kernel == "rbf":
                svc_kwargs["gamma"] = float(params["svc__gamma"])
            elif kernel == "poly":
                svc_kwargs["degree"] = int(params["svc__degree"])
            return mn, SVC(**svc_kwargs)

        if mn == "knn":
            return mn, KNeighborsClassifier(
                n_neighbors=int(params["knn__n_neighbors"]),
                weights=params["knn__weights"],
                p=int(params["knn__p"]),
                n_jobs=-1,
            )

        raise ValueError(f"Unknown model name in params: {mn!r}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_results(self) -> None:
        """Pretty-print a summary of the best trial."""
        if self.study_ is None or self.best_params_ is None or self.best_score_ is None:
            print("No study found. Run .fit() first.")
            return

        sep: str = "─" * 60
        print(f"\n{sep}")
        print("  OPTUNA CLASSIFICATION TUNING — RESULTS")
        print(sep)
        print(f"  Trials completed : {len(self.study_.trials)}")
        print(f"  Scoring metric   : {self.scoring}  ({self.direction})")
        print(f"  Best CV score    : {self.best_score_:.6f}")
        print("\n  Best parameters:")
        for k, v in sorted(self.best_params_.items()):
            print(f"    {k:<30s} = {v}")
        print(sep)

    def top_trials(self, n: int = 5) -> list[TrialSummary]:
        """
        Return the top-*n* trials sorted by score.

        Parameters
        ----------
        n:
            Number of trials to return (default ``5``).

        Returns
        -------
        list[TrialSummary]
            Each entry has typed keys ``rank``, ``score``, and ``params``.
        """
        if self.study_ is None:
            raise RuntimeError("Run .fit() first.")

        reverse: bool = self.direction == "maximize"
        completed: list[optuna.trial.FrozenTrial] = [
            t for t in self.study_.trials if t.value is not None
        ]
        ranked: list[optuna.trial.FrozenTrial] = sorted(
            completed,
            key=lambda t: t.value,  # type: ignore[return-value]
            reverse=reverse,
        )
        return [
            TrialSummary(rank=i + 1, score=float(t.value), params=t.params)  # type: ignore[arg-type]
            for i, t in enumerate(ranked[:n])
        ]

    def model_comparison(self) -> dict[ModelName, float]:
        """
        Return the best CV score achieved per model family.

        Returns
        -------
        dict[ModelName, float]
            Maps each model name that appeared in the study to its
            best observed CV score, sorted best-first.
        """
        if self.study_ is None:
            raise RuntimeError("Run .fit() first.")

        best: dict[ModelName, float] = {}
        reverse: bool = self.direction == "maximize"

        trial: optuna.trial.FrozenTrial
        for trial in self.study_.trials:
            if trial.value is None:
                continue
            mn: ModelName = trial.params.get("model", "unknown")  # type: ignore[assignment]
            score: float = float(trial.value)
            if mn not in best:
                best[mn] = score
            elif reverse:
                best[mn] = max(best[mn], score)
            else:
                best[mn] = min(best[mn], score)

        return dict(sorted(best.items(), key=lambda x: x[1], reverse=reverse))


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to csv file")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).dropna(axis=0)  ## TODO: make sure to make this part of the pipeline (imputation)
    X_all: NDArray[np.float64] = df.filter(regex="^feature").values
    y_all: NDArray[np.float64] = df["label"].values
    

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    tuner: ClassificationTuner = ClassificationTuner(
        n_trials=100,
        cv=5,
        scoring="f1_macro",
        verbose=True,
    )
    tuner.fit(X_train, y_train)

    assert tuner.best_pipeline_ is not None
    preds = tuner.best_pipeline_.predict(X_test)
    test_f1: float = f1_score(y_test, preds, average="macro")
    print(f"\n  Test F1 (macro): {test_f1:.4f}")

    print("\n  ── Model comparison (best CV score per family) ──")
    for model_name, score in tuner.model_comparison().items():
        print(f"    {model_name:<20s}  {score:.6f}")

    print("\n  ── Top 5 trials ──")
    summary: TrialSummary
    for summary in tuner.top_trials(5):
        print(
            f"  #{summary['rank']}  "
            f"score={summary['score']:.6f}  "
            f"model={summary['params'].get('model')}"
        )
