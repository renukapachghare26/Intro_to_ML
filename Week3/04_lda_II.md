# 04 – LDA II: Fisher's Criterion, Derivation and Multi-class

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

## Small multi-class intuition
Three groups of students: Low, Medium, High scorers, measured by (hours studied, attendance %).
- K = 3 gives at most 2 LDA axes.
- Axis 1 would mostly separate Low vs High (biggest gap).
- Axis 2 captures what remains, such as Medium vs the others.

## Edge cases
- If S_W is singular (more features than samples, or collinear features), the inverse fails.
  Fixes: use pseudo-inverse, add shrinkage (S_W + εI), or reduce dimensions with PCA first.
- Eigenvalues tell you **how much class separation each axis carries**.

## Key takeaways
- J(w) = (wᵀ S_B w) / (wᵀ S_W w), solved by an eigenproblem.
- Two classes: w ∝ S_W⁻¹(μ2 - μ1).
- K classes: at most K - 1 discriminant directions.