# 04 – LDA II: Gaussian Assumption, Derivation, Fisher's Criterion and Multi-class

## The extra assumption in LDA
- Already assumed: class-conditional density f_k(x) is **Gaussian**.
- LDA adds: **Σ_k = Σ** (the same covariance matrix for every class).
- Meaning:
  - 1D: each class's Gaussian can be **shifted** (different mean) but not **reshaped**.
  - 2D: the one-sigma contours must look alike for all classes (same size, shape and tilt). One class cannot be a circle while another is a long ellipse.
- Without this assumption you get **QDA** (see below).

## Deriving the boundary
- Boundary between classes k and l: where P(G=k | x) = P(G=l | x).
- So the ratio P(k | x) / P(l | x) = 1 and its **log = 0**. (Same trick as the log-odds in logistic regression.)
- Using Bayes' rule:

      log [ P(k|x) / P(l|x) ] = log [ f_k(x) π_k / f_l(x) π_l ]

- The denominator P(x) is common to both, so it **cancels**.
- With a shared Σ, many more terms cancel:
  - The normalizing constant (the 1/((2π)^(p/2)|Σ|^(1/2)) part) is the same → cancels.
  - The xᵀΣ⁻¹x (x² in 1D) terms are the same for both classes → **cancel**.
  - What is left: terms with x·μ and terms with μ².
- Result:

      log [ P(k|x) / P(l|x) ] = log(π_k/π_l) − ½ (μ_k + μ_l)ᵀ Σ⁻¹ (μ_k − μ_l) + xᵀ Σ⁻¹ (μ_k − μ_l)

- This is **linear in x**, so setting it to 0 gives a **hyperplane**.
- Linearity needs the equal-covariance assumption. If Σ_k differ, the x² term stays and the boundary is **quadratic → QDA**.
- Intuition on the sign: if x belongs to class k the numerator (class k) is larger; if class l, the denominator is larger. That tells you which side of the boundary x is on.

### K classes
- Boundaries are pairwise. For K classes, compare class by class: **K − 1 comparisons** to find the winner.
- Two classes → just one comparison.

## Discriminant function form of LDA
- Recall: classify x to class k if δ_k(x) > δ_l(x) for all l ≠ k.
- For LDA:

      δ_k(x) = xᵀ Σ⁻¹ μ_k − ½ μ_kᵀ Σ⁻¹ μ_k + log π_k

- Compute δ_k(x) for every class; the **largest** wins.
- (Note: Σ here is the covariance matrix, not a summation sign.)

## Estimating the parameters (from training data)
Three things must be estimated: **π, μ, Σ**.

1. **Priors**: π̂_k = N_k / N
   - N_k = number of training points in class k, N = total points.
   - Simple counting, but it is still estimated from data (not given).
2. **Class means**: μ̂_k = (1/N_k) Σ_{i in class k} x_i
   - Take all points of class k and find their centre.
3. **Shared covariance** (**pooled estimate**):

       Σ̂ = (1 / (N − K)) Σ_{k=1}^{K} Σ_{i: g_i = k} (x_i − μ̂_k)(x_i − μ̂_k)ᵀ

   - Uses data from **all classes**, not just one.
   - Subtract the **class mean μ̂_k** (not the overall mean). We only care about spread **within** each class, and training labels tell us each point's class.
   - Divide by **N − K** (not N − 1): the usual "−1" correction for one estimated mean becomes −K because K means were estimated. This gives an unbiased estimate.
   - More robust than computing each class's variance separately and then averaging.

- Recap: sample mean = Σx / n. Sample variance = Σ(x − mean)² / (n − 1) (unbiased because the mean depends on the data).
- Need a reasonable amount of data per class. With very little data most estimation methods struggle (SVM is one method that copes with little data, covered later).

## Geometry of the LDA boundary
- If the class covariance were **spherical** (circular contours), the boundary would be **perpendicular to the line joining the means**.
- If the covariance is slanted/elliptical, the boundary is also **at an angle** to the line joining the means.
- Example: with 3 classes, you get 3 pairwise linear boundaries meeting each other.

## LDA as a direction-finding / feature-reduction method
- Many pattern recognition books present LDA as **feature selection / dimensionality reduction**.
- Analogy with regression:
  - **PCR** (principal component regression) looks only at the inputs: directions of max input variance.
  - **PLS** also takes the **response** into account.
  - **LDA** is the classification counterpart: it uses the **class labels**.
- **PCA** maximizes the variance of the data (ignores labels).
- **LDA** finds directions where the **between-class variance is maximized** and the **within-class variance is minimized**.
  - The class means are spread as far apart as possible along the chosen direction.
- This is the idea behind **Fisher's criterion** below.

## Fisher's criterion
After projecting onto w:
- Projected means: m_k = wᵀμ_k
- Projected scatter of class k: s_k² = Σ (wᵀx - m_k)²

Fisher's objective:

    J(w) = (m2 - m1)² / (s1² + s2²)  =  (wᵀ S_B w) / (wᵀ S_W w)

Goal: **maximize J(w)**.
(Top = separation of means. Bottom = spread within classes.)

## Solving it
Take the derivative of J(w) with respect to w and set it to zero:

    S_B w = λ S_W w          (a generalized eigenvalue problem)
    →  S_W⁻¹ S_B w = λ w

For **two classes**, S_B w always points in the direction (μ2 - μ1), so:

    w ∝ S_W⁻¹ (μ2 - μ1)

(We only need the direction, so the scale of w does not matter.)

## Continuing the example from note 03
- w ∝ (0.427, 0.196). Normalized: w ≈ (0.909, 0.417)
- Projected means (using the unnormalized w): 2.03 and 5.08. The gap is about 3.05.
- Projected spread is small compared with that gap, so J(w) is large. A good split.

Compare a bad direction, say w = (0, 1) (just the y-axis).
Class 1 y-values: 2,4,3,6,4 and Class 2 y-values: 10,8,5,7,8 → they overlap a bit (5 vs 6),
so the separation is worse. Fisher's w avoids this.

## Multi-class LDA (K classes, d features)
Definitions:
- Overall mean μ
- S_W = Σ_k S_k (sum of within-class scatters)
- S_B = Σ_k N_k (μ_k - μ)(μ_k - μ)ᵀ  (N_k = number of points in class k)

Solve S_W⁻¹ S_B w = λ w and take the eigenvectors with the **largest eigenvalues**.

**Number of useful directions = min(K - 1, d)**
- 2 classes → 1 direction
- 3 classes → at most 2 directions (so you can plot 3 classes in 2D)
- 10 classes (e.g., digits 0–9) → at most 9 directions

## Step-by-step recipe (for exams / coding)
1. Compute the class means μ_k and the overall mean μ.
2. Compute S_W and S_B.
3. Compute the eigenvectors/eigenvalues of S_W⁻¹ S_B.
4. Sort by eigenvalue (descending) and keep the top r eigenvectors → matrix W.
5. Project: Z = X W.
6. Classify in the projected space (nearest class mean or a threshold).

### Recipe for the Gaussian/Bayes view of LDA
1. Estimate π̂_k, μ̂_k, and the pooled Σ̂ from training data.
2. For a new x compute δ_k(x) for every class.
3. Predict the class with the largest δ_k(x).

## Small multi-class intuition
Three groups of students: Low, Medium, High scorers, measured by (hours studied, attendance %).
- K = 3 gives at most 2 LDA axes.
- Axis 1 would mostly separate Low vs High (biggest gap).
- Axis 2 captures what remains, such as Medium vs the others.

## Edge cases
- If S_W is singular (more features than samples, or collinear features), the inverse fails.
  Fixes: use pseudo-inverse, add shrinkage (S_W + εI), or reduce dimensions with PCA first.
- Eigenvalues tell you **how much class separation each axis carries**.
- Very small training sets make the estimates of μ, Σ and π unreliable.
- Unequal covariances across classes → the shared-Σ assumption breaks → use QDA.

## Key takeaways
- LDA = Gaussian class densities + **one shared covariance Σ**.
- Boundary from log-ratio of posteriors = 0; x² terms cancel, so the boundary is **linear**. Without the shared Σ you get **QDA** (quadratic).
- Discriminant: δ_k(x) = xᵀΣ⁻¹μ_k − ½μ_kᵀΣ⁻¹μ_k + log π_k; largest wins.
- Estimates: π̂_k = N_k/N, μ̂_k = class mean, Σ̂ = **pooled** within-class covariance divided by N − K.
- Spherical covariance → boundary perpendicular to the line joining the means; otherwise slanted.
- LDA is the classification analogue of PLS: it uses labels to find directions that spread class means apart while keeping within-class spread small.
- J(w) = (wᵀ S_B w) / (wᵀ S_W w), solved by an eigenproblem.
- Two classes: w ∝ S_W⁻¹(μ2 - μ1).
- K classes: at most K - 1 discriminant directions.