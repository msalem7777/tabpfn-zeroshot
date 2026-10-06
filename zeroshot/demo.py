"""A reproducible fictional example; no invented research is presented as real."""
import numpy as np
import pandas as pd

from .evidence import source_record


def demo_bundle():
    rng = np.random.default_rng(107)
    frame = pd.DataFrame({"temperature": rng.uniform(15, 35, 60),
                          "pressure": rng.uniform(1, 3, 60)})
    quote = "Fictional demonstration: yield = 20 + 1.5 * temperature + 4 * pressure. Residual SD is 3. Coefficient SEs are 1, 0.1, 0.4."
    source = source_record("Synthetic example — NOT a published study", quote, kind="synthetic")
    predictors = [{"column": name, "meaning": name, "input_unit": unit, "study_unit": unit,
                   "multiplier": 1, "offset": 0, "transform": "identity", "source_center": 0,
                   "source_scale": 1, "category_map": None}
                  for name, unit in [("temperature", "Celsius"), ("pressure", "bar")]]
    model = {"id": "synthetic-1", "title": source["title"], "family": "linear",
             "target": "yield", "target_definition": "Fictional process yield", "target_unit": "units",
             "source_population": "Fictional demonstration process", "positive_class": None,
             "predictors": predictors, "coefficients": [20, 1.5, 4], "covariance": None,
             "standard_errors": [1, 0.1, 0.4], "independent_coefficients": True,
             "parameter_distribution": "multivariate_normal", "residual_sd": 3, "residual_log_sd": 0,
             "target_center": 0, "target_scale": 1, "target_transform": "identity",
             "transfer_intercept_sd": 0, "weight": 1, "weight_reason": "Only fictional candidate",
             "uncertainty_basis": "Demo normal parameter distribution; not inferred from research",
             "assumptions": ["Independent normal coefficients", "Gaussian residuals with fixed SD 3",
                             "No source-to-target shift", "Uploaded and source units coincide"],
             "anchors": [{"source_id": source["id"], "quote": quote, "supports": "Entire fictional model"}],
             "reviewed": False, "synthetic": True}
    target = {"name": "yield", "definition": "Fictional process yield", "unit": "units",
              "population": "Fictional demonstration process", "time_horizon": "same observation",
              "family": "linear", "positive_class": None}
    return frame, {"target": target, "models": [model], "sources": [source]}
