# Statistical specification and scope

This describes the current implementation, not a claim that its approximations are statistically identified from unlabeled data. Read the short overview in [README](../README.md) first.

## 1. Transfer through synthetic supervision

Inputs are context covariates $X_C$, test covariates $X_T$, a target definition, source material, and explicit assumptions. The desired conditional distribution $p(y\mid x)$ is unavailable. External equations supply an assumed family of conditional relationships; TabPFN supplies a separate pretrained prior over tabular prediction problems. The method transfers information through generated labels. It does not update neural weights or fit coefficients against masked target labels.

Conceptually, for a label-generating distribution $\pi$:

$$
q_{\mathrm{mix}}(y_{\ast}\mid x_{\ast},X_C,E)
=\int q_{\mathrm{PFN}}(y_{\ast}\mid x_{\ast},X_C,\tilde{y}_C)
\,p(d\tilde{y}_C\mid X_C,J,\theta)\,\pi(dJ,d\theta\mid E).
$$

Monte Carlo approximates this integral. Writing $\pi(\cdot\mid E)$ does not imply a posterior inferred by Bayes' rule: it may be a distribution elicited from an LLM or manually declared. Regression mean-label mode replaces the outcome distribution with a point mass at each sampled equation's mean.

## 2. Equation scale and parameter uncertainty

A linear predictor is $\eta_i=\beta_0+\sum_j\beta_j\phi_j(x_i)$, where every $\phi_j$ must match the source's definition, coding, unit and transformation. For a published standardized predictor, $\phi_j(x)=(x_j-a_j)/s_j$ must use the relevant declared center $a_j$ and scale $s_j$. Standardizing target-domain columns arbitrarily does not preserve published coefficients. A missing intercept, incompatible time horizon, or incompatible outcome is not repaired by normalization alone.

Supported coefficient uncertainty includes:

$$
\beta^{(m)}\sim\mathcal N(\hat\beta,\Sigma)
\quad\text{or}\quad
\beta^{(m)}=b_{K_m},\; K_m\sim\mathrm{Uniform}\{1,\ldots,B\}.
$$

The latter samples complete supplied bootstrap/posterior coefficient vectors, preserving dependence within a vector. The engine does not reconstruct a raw-data bootstrap from an article. Paired residual draws can retain their supplied dependence. Diagonal covariance $\Sigma=\operatorname{diag}(SE_j^2)$ requires an explicit coefficient-independence assumption. Coefficient standard errors are not outcome standard deviations. An SD of a covariate is not a coefficient SE. Converting a symmetric interval to an SE assumes the relevant sampling distribution; it is not evidence of an actual posterior sample.

The engine can sample residual log standard deviation and an additional intercept-shift scenario when declared. These distributions do not automatically cover population shift or publication bias. Proposed uncertainty ranges remain assumptions.

## 3. Context labels in each world

For binary targets:

$$
p_i^{(m)}=\frac{1}{1+\exp(-\eta_i^{(m)})},\qquad
\tilde y_i^{(m)}\sim\mathrm{Bernoulli}(p_i^{(m)}).
$$

Hard 0/1 labels are used by the classifier. For Gaussian regression in mean mode, $\tilde y_i^{(m)}=\eta_i^{(m)}$; in optional outcome mode, $\tilde y_i^{(m)}=\eta_i^{(m)}+\epsilon_i^{(m)}$, with declared residual noise. If the source models log outcomes with Gaussian residuals, the original-scale conditional mean is $\exp(\eta_i+\sigma^2/2)$, not simply $\exp(\eta_i)$.

One parameter vector per participating model/world is shared across rows, preserving parameter-driven co-movement. Context covariates and selected row subsets/folds stay fixed across worlds; generated labels change. Bernoulli outcomes or declared missing inputs add row-level variability.

Candidate equations carry nonnegative declared scenario weights $w_k$. With identical eligibility across rows, a shared model draw selects the equation for that world. If eligibility differs, weights are renormalized over each row's eligible models and a shared uniform variate selects by each row's inverse CDF. This coupling is an implementation convention, not an inferred joint distribution. It can give different rows different eligible models in the same world.

## 4. Missing and extra covariates

When a source uses an unavailable predictor $Z$, valid marginalization requires an assumed distribution:

$$
p(y\mid x)=\int p(y\mid x,z)\,p(z\mid x)\,dz.
$$

The implementation supports declared normal, lognormal and categorical missing-predictor distributions, with optional conditional means. It does not learn a general joint distribution over missing variables. Multiple missing predictors need explicit support for the independence approximation. Replacing $Z$ by its mean is generally not equivalent to integrating a nonlinear logistic model. Deleting $\beta_Z Z$ also does not recover a correctly refitted reduced logistic model. A published reduced equation or an explicitly labeled proposed reduced model is a distinct candidate.

Conversely, if a context column has no effect in a source equation, there is no sourced coefficient to sample or marginalize. It may still be a TabPFN input, but any resulting relationship comes from the generated labels, feature correlations and TabPFN's prior. The column does not acquire an independently evidenced target effect.

Unsupported context rows are excluded from label generation. The primary path retains test rows without requiring a literature equation evaluable on those rows; selected TabPFN feature columns must be present. Without a separate test set, held-out folds prevent a row's own generated label from being used to predict that row. This is a leakage precaution, not validation against reality.

## 5. Aggregation and interpretation

Each world yields a predictive distribution $q_m$. The estimator is $\hat q=M^{-1}\sum_m q_m$. Let its component mean/variance be $\mu_m,v_m$. By the law of total variance:

$$
\mathrm{Var}_{\hat{q}}(Y)=\frac{1}{M}\sum_m v_m+\frac{1}{M}\sum_m(\mu_m-\bar\mu)^2.
$$

The second term uses denominator $M$ because this is the variance of the finite mixture. Do not divide the predictive variance by $M$ to report outcome uncertainty. Under suitable independent sampling, Monte Carlo error in a mean decreases approximately as $M^{-1/2}$; that is numerical precision, not narrower uncertainty about an individual's outcome.

For binary predictions, $q_m=\mathrm{Bernoulli}(p_m)$ and:

$$
\bar p=\frac1M\sum_m p_m,\qquad
\bar p(1-\bar p)=\frac1M\sum_m p_m(1-p_m)+\frac1M\sum_m(p_m-\bar p)^2.
$$

The outcome variance and an interval across world probabilities answer different questions. An interval across $p_m$ expresses sensitivity to the assumed worlds; it is not automatically a calibrated confidence/credible interval for the true risk. For regression, output quantiles are estimated from pooled component predictive draws, including the adapter's full-support tail treatment. Finite draws add numerical approximation error.

Neither component should automatically be called pure aleatoric or epistemic uncertainty: TabPFN and the label generator mix assumptions at different levels. Their composition may double-count or transform uncertainty. An exact Bayesian claim would require a compatible joint generative model and a justified inference procedure, which this prototype does not establish.

## 6. What would demonstrate value?

Evaluate frozen choices on untouched target labels and multiple external datasets. Compare the full pipeline against direct sampled-equation predictions, simple synthetic-label learners, fixed-prior baselines, and supervised comparators where appropriate. Repeat splits/seeds and test sensitivity to model weights, parameter spreads, context size and world count. Assess discrimination, proper probability scores and calibration; assess regression predictive coverage separately. Do not tune against the final evaluation set. Source exclusion cannot rule out pretraining contamination, and neither prediction accuracy nor source citations establish causality or fairness.
