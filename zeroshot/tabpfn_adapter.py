"""Optional real TabPFN inference. No weight training and no imitation fallback."""
import gc

import numpy as np

from .distributions import bar_samples


class TabPFNAdapter:
    def __init__(self, family, config):
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError:
            raise ValueError('TabPFN is not installed. Run: python -m pip install -e ".[tabpfn]"') from None
        cls = TabPFNClassifier if family == "logistic" else TabPFNRegressor
        kwargs = dict(device=config.device, n_estimators=config.estimators,
                      random_state=config.seed, fit_mode="low_memory")
        if config.model_path:
            kwargs["model_path"] = config.model_path
        self.model = cls(**kwargs)
        self.family = family

    def predict(self, x_context, y_context, x_test, count, rng):
        # One model is reused sequentially. Previous contexts are replaced by fit.
        if len(np.unique(y_context)) < 2:
            raise ValueError("A sampled TabPFN context has only one target value/class. Increase context size or revise evidence; this world was not silently dropped.")
        self.model.fit(x_context, y_context)
        if self.family == "logistic":
            classes = list(self.model.classes_)
            if 1 not in classes:
                raise ValueError("Class 1 is absent from the fitted classifier.")
            p = np.asarray(self.model.predict_proba(x_test))[:, classes.index(1)]
            return p, p*(1-p), rng.binomial(1, p, size=(count, len(p)))
        full = self.model.predict(x_test, output_type="full")
        if not {"criterion", "logits"} <= full.keys():
            raise ValueError("Installed TabPFN does not expose the supported full-distribution contract.")
        criterion, logits = full["criterion"], full["logits"]
        mean = criterion.mean(logits).detach().cpu().numpy().astype(float)
        variance = criterion.variance(logits).detach().cpu().numpy().astype(float)
        if np.min(variance) < -1e-4:
            raise ValueError("TabPFN produced materially negative variance.")
        samples = bar_samples(criterion, logits, count, rng)
        return mean, np.maximum(variance, 0), samples

    def close(self):
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

