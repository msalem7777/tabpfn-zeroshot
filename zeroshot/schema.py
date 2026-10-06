"""Strict execution contract. Incomplete LLM drafts remain editable dictionaries."""
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Predictor(StrictModel):
    column: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    input_unit: str = Field(min_length=1)
    study_unit: str = Field(min_length=1)
    # Convert uploaded units, transform, then apply SOURCE centering/scaling.
    multiplier: float
    offset: float
    transform: Literal["identity", "log", "log1p", "square"]
    source_center: float
    source_scale: float = Field(gt=0)
    category_map: dict[str, float] | None = None


class Anchor(StrictModel):
    source_id: str
    quote: str = Field(min_length=12)
    supports: str = Field(min_length=1)


class MissingDistribution(StrictModel):
    """Source-supported distribution in the predictor's INPUT units.

    For normal/lognormal distributions, conditional_mean_coefficients maps
    observed numeric columns to slopes; mean is the intercept. For lognormal,
    mean and sd refer to log(input value). Categorical values are raw codes.
    These distribution parameters are treated as fixed reported quantities.
    """
    column: str = Field(min_length=1)
    family: Literal["normal", "lognormal", "categorical"]
    mean: float | None = None
    sd: float | None = Field(default=None, ge=0)
    conditional_mean_coefficients: dict[str, float] = Field(default_factory=dict)
    values: list[float | str] = Field(default_factory=list)
    probabilities: list[float] = Field(default_factory=list)
    basis: str = Field(min_length=1)
    # Empty means that joint independence has NOT been established. Multiple
    # missing variables in one row require source-backed conditional independence.
    independence_basis: str = ""
    anchors: list[Anchor] = Field(min_length=1)

    @model_validator(mode="after")
    def check_distribution(self):
        if self.column in self.conditional_mean_coefficients:
            raise ValueError("A missing predictor cannot condition on itself.")
        if self.family == "categorical":
            if (not self.values or len(self.values) != len(self.probabilities)
                    or min(self.probabilities) < 0 or not np.isclose(sum(self.probabilities), 1)
                    or len(set(map(str, self.values))) != len(self.values)):
                raise ValueError("Categorical support needs unique values and probabilities summing to one.")
            if self.conditional_mean_coefficients or self.mean is not None or self.sd is not None:
                raise ValueError("Categorical distributions use values/probabilities only.")
        elif self.mean is None or self.sd is None or self.values or self.probabilities:
            raise ValueError("Normal/lognormal distributions need reported mean and sd only.")
        return self


class EvidenceModel(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    family: Literal["linear", "logistic"]
    target: str = Field(min_length=1)
    target_definition: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    source_population: str = Field(min_length=1)
    positive_class: str | None = None
    predictors: list[Predictor] = Field(min_length=1)
    missing_distributions: list[MissingDistribution] = Field(default_factory=list)
    # Coefficient order is ALWAYS intercept, then predictors in listed order.
    coefficients: list[float]
    covariance: list[list[float]] | None = None
    standard_errors: list[float] | None = None
    independent_coefficients: bool = False
    parameter_distribution: Literal["multivariate_normal", "empirical"]
    # Each row is one complete reported bootstrap/posterior coefficient vector.
    parameter_draws: list[list[float]] | None = None
    residual_draws: list[float] | None = None
    # sigma is in the model's target scale; zero explicitly means no residual noise.
    residual_sd: float = Field(ge=0)
    residual_log_sd: float = Field(ge=0)
    target_center: float
    target_scale: float = Field(gt=0)
    target_transform: Literal["identity", "log"]
    # Independent extra intercept uncertainty represents an APPROVED transfer scenario.
    transfer_intercept_sd: float = Field(ge=0)
    weight: float = Field(gt=0)
    weight_reason: str = Field(min_length=1)
    uncertainty_basis: str = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    anchors: list[Anchor] = Field(default_factory=list)
    reviewed: bool = False
    synthetic: bool = False
    # Proposed models are explicit assumptions, not claimed published fits.
    model_origin: Literal['published', 'llm_proposed', 'user_specified'] = 'published'
    proposal_basis: str = ''

    @model_validator(mode="after")
    def check_parameters(self):
        if self.model_origin == 'published' and not self.anchors:
            raise ValueError('Published models require supporting source quotations.')
        if self.model_origin != 'published' and not self.proposal_basis.strip():
            raise ValueError('A proposed model must explain its assumptions and their basis.')
        n = len(self.predictors) + 1
        names = [d.column for d in self.missing_distributions]
        if len(set(names)) != len(names) or not set(names) <= {p.column for p in self.predictors}:
            raise ValueError("Missing distributions must refer uniquely to required predictor columns.")
        if len(self.coefficients) != n:
            raise ValueError("Coefficients must contain intercept followed by one slope per predictor.")
        if self.parameter_distribution == "empirical":
            draws = np.asarray(self.parameter_draws, dtype=float)
            if draws.ndim != 2 or draws.shape[1] != n or len(draws) < 2 or not np.isfinite(draws).all():
                raise ValueError("Empirical uncertainty needs at least two complete, finite coefficient vectors.")
            if self.covariance is not None or self.standard_errors is not None:
                raise ValueError("Empirical draws replace Gaussian covariance/standard errors.")
            if self.residual_draws is not None:
                if len(self.residual_draws) != len(draws) or min(self.residual_draws) < 0 or self.residual_log_sd:
                    raise ValueError("Residual draws must align with coefficient draws and replace log-SD uncertainty.")
                if self.family == "logistic" and any(self.residual_draws):
                    raise ValueError("Logistic models cannot have Gaussian residual draws.")
        elif self.parameter_draws is not None or self.residual_draws is not None:
            raise ValueError("Parameter/residual draws require empirical uncertainty.")
        elif self.covariance is not None:
            a = np.asarray(self.covariance, dtype=float)
            if a.shape != (n, n) or not np.allclose(a, a.T, atol=1e-10):
                raise ValueError("Covariance must be square and symmetric in coefficient order.")
            if np.linalg.eigvalsh(a).min() < -1e-10:
                raise ValueError("Covariance must be positive semidefinite (a valid uncertainty matrix).")
            if self.standard_errors is not None:
                raise ValueError("Supply covariance OR standard_errors, not both.")
        else:
            if not self.independent_coefficients or self.standard_errors is None:
                raise ValueError("Missing covariance: supply standard_errors and explicitly approve independent_coefficients.")
            if len(self.standard_errors) != n or min(self.standard_errors) < 0:
                raise ValueError("One nonnegative standard error is required per coefficient.")
        if self.family == "logistic":
            if not self.positive_class:
                raise ValueError("Define what class 1 means.")
            if (self.residual_sd or self.residual_log_sd or self.target_center != 0
                    or self.target_scale != 1 or self.target_transform != "identity"):
                raise ValueError("Logistic models use Bernoulli outcomes, no added Gaussian noise or target rescaling.")
        if self.residual_sd == 0 and self.residual_log_sd:
            raise ValueError("A zero residual scale cannot have log-scale uncertainty.")
        return self

    def coefficient_covariance(self):
        if self.covariance is not None:
            return np.asarray(self.covariance)
        return np.diag(np.square(self.standard_errors))


class Target(StrictModel):
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    population: str = Field(min_length=1)
    time_horizon: str = Field(min_length=1)
    family: Literal["linear", "logistic"]
    positive_class: str | None = None


class RunConfig(StrictModel):
    method: Literal["tabpfn", "direct", "both"] = "tabpfn"
    regression_context_labels: Literal["mean", "outcome"] = "mean"
    worlds: int = Field(default=32, ge=2, le=1000)
    samples_per_world: int = Field(default=128, ge=16, le=2048)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    interval: float = Field(default=0.9, gt=0, lt=1)
    context_limit: int = Field(default=512, ge=8, le=10000)
    folds: int = Field(default=3, ge=2, le=10)
    device: Literal["cpu", "cuda", "auto"] = "auto"
    estimators: int = Field(default=1, ge=1, le=8)
    model_path: str | None = None
    use_all_features: bool = True
    allow_synthetic: bool = False
    assumptions_approved: bool = False
