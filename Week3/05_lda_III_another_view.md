# 05 – LDA III: Another View (Fisher Derivation and Probabilistic / Bayes View)

There are two ways to arrive at LDA:
1. **Geometric** (Fisher): maximize between-class variance relative to within-class variance. No Gaussian assumption.
2. **Probabilistic** (Bayes): Gaussian classes with a shared covariance.

This note derives the first one step by step, then the second, then shows they give the **same direction**.

(Reading: for the Fisher part see PRML by Bishop. For the Gaussian/Bayes part see ESL by Hastie, Tibshirani and Friedman.)

---

# Part A: Fisher's derivation from scratch

## Between-class and within-class variance
- **Between-class variance** = variance of the (projected) **class means**.
  - 2 classes → just the **distance** between the two projected means.
  - K classes → the spread (variance) among the K projected centres.
- **Within-class variance** = for each class, the variance of projected points around that class's projected mean (computed per class, then added).
- Goal: maximize between-class variance **relative to** within-class variance.

## Two-class setup
- Project: y = wᵀx.
- Decision rule: y > w0 → class 1; y ≤ w0 → class 2. (w0 = threshold.)
- Class means in the original space: μ1, μ2.
- **Projected** means: m_k = wᵀμ_k.
- (Notation: the lecture uses a bar for the original means and no bar for projected ones, because bold is hard on a board. Textbooks use bold for the original mean and plain for the projection.)

## Step 1: only the between-class criterion
- Maximize the distance between the projected means:

      m2 − m1 = wᵀ(μ2 − μ1)

- **Problem**: unbounded. Scale w up and the value grows without limit.
- Fix: constrain **‖w‖ = 1** (a common trick to avoid unbounded solutions).
- Could use ‖w‖² ≤ 1 instead, but since we are maximizing, the solution scales up until it hits 1 anyway, so it is the same as ‖w‖ = 1.
- Solve with a constraint (Lagrange) → **w ∝ (μ2 − μ1)**.
- Meaning: project everything onto the line joining the two means, then pick a threshold w0.
  - If the classes are **spherical** (equal circular spread), the threshold is the **midpoint** of the line joining the means.
  - Otherwise the threshold lies somewhere else along the line.

### What goes wrong (the missing piece)
- Picture two Gaussians with one-sigma contours. Training points are mixed: some "−" points appear inside class 1's region and some "+" in class 2's, because a Gaussian extends beyond its contour (the contour is only the most probable region, not a hard limit).
- Projecting onto the line joining the means can leave the projected classes **overlapping**.
- Missing: the **within-class variance**. That is what we add next.

## Step 2: add within-class variance
- Projected scatter of class k:

      s_k² = Σ_{i in class k} (wᵀx_i − m_k)²

- Total within-class: s1² + s2² (not divided by the number of points; constants do not matter for the maximization).

## Fisher's criterion
Named after **Fisher**, the statistician who came up with LDA decades ago.

    J(w) = (m2 − m1)² / (s1² + s2²)

### Rewrite in matrix form
- Numerator: (m2 − m1)² = (wᵀμ2 − wᵀμ1)² = wᵀ (μ2 − μ1)(μ2 − μ1)ᵀ w = **wᵀ S_B w**
  - **S_B = (μ2 − μ1)(μ2 − μ1)ᵀ** is the between-class matrix.
- Denominator: s1² + s2² = **wᵀ S_W w**
  - S_W = S_1 + S_2, with S_k = Σ (x − μ_k)(x − μ_k)ᵀ (pull w out of each class's term).
- So:

      J(w) = (wᵀ S_B w) / (wᵀ S_W w)

## Maximizing J(w)
1. It is a ratio u/v. Differentiate with respect to w and set to 0.
2. At the optimum the "denominator squared" part drops out, and we equate the two halves of the numerator:

       (wᵀ S_B w) · S_W w = (wᵀ S_W w) · S_B w

3. Quadratics like wᵀAw differentiate to something linear in w. (Tip: write the matrix form out in detail, differentiate term by term, then see the pattern.)
4. Key fact: **S_B w is always in the direction of (μ2 − μ1)**. (We already saw the same direction when we used only the between-class criterion.)
5. The bracketed quantities wᵀS_Bw and wᵀS_Ww are just **scalars**, so they only affect scale.
6. Result (proportional, not equal):

       w ∝ S_W⁻¹ (μ2 − μ1)

- Compare:
  - Between-class only: w ∝ (μ2 − μ1).
  - Adding within-class variance: multiply by **S_W⁻¹**, which corrects for the shape of the within-class spread.

## Same direction as the Gaussian view
- Σ (the shared covariance in the Gaussian LDA) plays the role of S_W (within-class covariance).
- Gaussian LDA boundary direction: Σ⁻¹ (μ_k − μ_l).
- Fisher direction: S_W⁻¹ (μ2 − μ1).
- **Same direction**, up to scaling and non-x constants.
- You can motivate LDA either way:
  - Start with Gaussian class densities and find the boundary, or
  - Start with "between-class vs within-class variance" as an objective.

## Important consequence: no Gaussian needed
- The Fisher derivation uses only **sample means and sample variances**.
- No assumption about the class-conditional distribution was made.
- So LDA still has a well-defined meaning (finding a good separating direction) **even when the data are not Gaussian**.
- The Gaussian assumption is needed only for the probabilistic interpretation (posteriors, priors shift, etc.).

---

# Part B: Probabilistic (Bayes) view

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

Example in words: a point at x = 4.2 is slightly nearer to Class 2's mean (distance 1.8 vs 2.2),
but since Class 1 is 4× more common, we still call it Class 1 (4.2 < 4.35).

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
- Fisher's view (geometry) and the Bayes view (probability) give the **same linear boundary direction** under Gaussian + equal covariance assumptions.
- Fisher's criterion: J(w) = (wᵀ S_B w)/(wᵀ S_W w) = between-class variance / within-class variance.
- Between-class term alone gives w ∝ (μ2 − μ1); adding within-class variance gives **w ∝ S_W⁻¹(μ2 − μ1)**.
- A constraint such as ‖w‖ = 1 is needed, otherwise scaling w makes the objective unbounded.
- The Fisher derivation needs **no Gaussian assumption**, so LDA is meaningful even for non-Gaussian data.
- Priors shift the boundary toward the rarer class.
- LDA is generative; logistic regression is discriminative.
- Relaxing the shared covariance gives QDA.