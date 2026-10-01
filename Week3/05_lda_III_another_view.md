# 05 – LDA III: Another View (Probabilistic / Bayes View)

In note 04 we derived LDA from a **geometric** idea (Fisher's separation).
There is a second, **probabilistic** way to arrive at the same classifier.

## Setup
Assume each class k generates data from a Gaussian with:
- its own mean μ_k
- a **shared covariance** Σ
- prior probability π_k (how common class k is)

Bayes' rule: pick the class with the highest posterior P(k | x) ∝ π_k * N(x | μ_k, Σ)

## The discriminant function
Taking the log and dropping terms that are the same for every class gives:

    δ_k(x) = xᵀ Σ⁻¹ μ_k  -  ½ μ_kᵀ Σ⁻¹ μ_k  +  ln π_k

Predict the class with the **largest δ_k(x)**.
This is **linear in x**, which is why it is called *Linear* Discriminant Analysis.
(If each class had its own Σ_k, the x² terms would not cancel and we would get QDA, a quadratic boundary.)

## 1D example (easy to see)
Two classes, same variance σ² = 1:
- Class 1 mean μ1 = 2, Class 2 mean μ2 = 6

**Equal priors (0.5 each):**
Boundary = midpoint = (2 + 6)/2 = **4**

**Unequal priors: π1 = 0.8, π2 = 0.2:**

    boundary = (μ1 + μ2)/2 + σ² * ln(π2/π1) / (μ1 - μ2)
             = 4 + 1 * ln(0.25) / (-4)
             = 4 + 0.347 = **4.35**

The boundary moved **toward the rarer class** (Class 2), which makes sense:
Class 1 is more common, so it gets a slightly bigger share of the space.

Example in words: a point at x = 4.2 is closer to... both, but since Class 1 is 4× more common,
we call it Class 1 (4.2 < 4.35).

## Link to logistic regression
For two classes, the LDA posterior is exactly a sigmoid of a linear function:

    P(class 2 | x) = σ( wᵀx + b )

with w = Σ⁻¹ (μ2 - μ1).
So **LDA and logistic regression have the same form** but are trained differently:

| | LDA | Logistic regression |
|---|---|---|
| Type | Generative (models P(x|class)) | Discriminative (models P(class|x)) |
| Assumes | Gaussian classes, shared covariance | Nothing about the distribution of x |
| Works best | Small data, assumptions hold | Larger data, assumptions violated |
| Outliers | More sensitive | More robust |

## Another view: nearest class mean
With equal priors and Σ = I (identity), the rule becomes: **assign x to the class with the nearest mean**.
With a general Σ, it is the nearest mean using **Mahalanobis distance**, which takes spread and correlation into account.

## Another view: least squares
For two classes, if you do linear regression on targets (e.g., -1 / +1), the resulting direction
is the same as the LDA direction (the threshold may differ).

## Dimensionality reduction view
Project onto the top K - 1 discriminant axes and classify there.
This gives a low-dimensional plot where classes are well separated, for example Iris (4 features → 2 axes).

## Key takeaways
- Fisher's view (geometry) and the Bayes view (probability) give the **same linear boundary** under Gaussian + equal covariance assumptions.
- Priors shift the boundary toward the rarer class.
- LDA is generative; logistic regression is discriminative.
- Relaxing the shared covariance gives QDA.