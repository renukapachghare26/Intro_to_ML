# Support Vector Machines I: Formulation

## Recap
- Hyperplane: f(x) = wᵀx + b = 0. (Lecture uses β, β0; here w = β, b = β0.)
- **f(x) gives the signed distance** to the hyperplane (exact when ‖w‖ = 1, otherwise f(x)/‖w‖).
- Perceptron problems:
  - Separable data → converges, but possibly slowly.
  - Not separable → cycles.
  - Even when it converges, **where** it converges depends on the starting point. No particular solution is singled out.
- Goal now: characterize **one specific optimal separating hyperplane**.
- Starting case: data is **linearly separable** (a hyperplane exists that splits the classes perfectly).

## What should "optimal" mean?
- Idea 1 (suggested in class): maximize the **sum of distances** of all points from the hyperplane.
- Better idea: **maximize the distance of the closest point** to the hyperplane.
  - The closest point from **each class** should be at the **same distance** from the hyperplane.
  - Then choose the orientation of the hyperplane so that this distance is as large as possible.
- Picture: a **slab** around the hyperplane with no points inside it, equally thick on both sides.

## Margin
- The empty slab around the hyperplane is the **margin**.
- Margin = distance of the closest point to the hyperplane.
- Classifiers built this way are called **max-margin classifiers** (optimal hyperplane classifiers).

## Step 1: first formulation
- y_i ∈ {+1, −1}.
- y_i (x_iᵀw + b) = signed distance of x_i, multiplied by y_i so it is **positive** when x_i is on the correct side. (Valid when ‖w‖ = 1.)
- Require **every** training point to be at least M away:

      maximize M  over  w, b, ‖w‖ = 1
      subject to  y_i (x_iᵀw + b) ≥ M,   for i = 1, ..., N

- Why this works:
  - Every point is at least M from the hyperplane. Maximizing M pushes the closest point as far away as possible.
  - The optimal M **is** the distance of the closest point, i.e. the margin.
  - M cannot be made arbitrarily large, because a w satisfying all N constraints may not exist.
- ‖w‖ = 1 is needed so the solution does not blow up (otherwise just scale w to beat M).

## Step 2: remove the ‖w‖ = 1 constraint
- Put the norm in the denominator of the constraint instead:

      maximize M  over  w, b
      subject to  (1/‖w‖) · y_i (x_iᵀw + b) ≥ M,   i = 1..N

- Same problem, just algebra (and geometry): y_i f(x_i)/‖w‖ is exactly the signed distance of any w.

## Step 3: fix the scale of w
- If some w satisfies the constraints, then **any scaled version** of it does too (‖w‖ appears on both sides after scaling).
- So only the **direction** of w matters. We are free to choose its scale.
- Choose:

      ‖w‖ = 1 / M

- Then the constraint becomes:

      y_i (x_iᵀw + b) ≥ 1,   i = 1..N

- No real change was made. It is just a normalization of the same problem.

## Step 4: change the objective
- Since ‖w‖ = 1/M, **maximizing M = minimizing ‖w‖**.
- Minimizing ‖w‖ is the same as minimizing ‖w‖² (norms are non-negative, so squaring does not change the minimizer).
- Squared form is easier to differentiate. (A factor of ½ is usually added for convenience; it does not change the answer.)

## Final formulation (hard-margin SVM, separable case)

      minimize   ½ ‖w‖²
      over       w, b
      subject to y_i (x_iᵀw + b) ≥ 1,   i = 1..N

- This is the **max-margin / optimal separating hyperplane** problem.
- Interpretation: find the **minimum-norm** w such that every data point is correctly classified **with room to spare**.

## What the constraint means
- y_i = +1: need wᵀx_i + b ≥ +1 (positive side, at least 1).
- y_i = −1: need wᵀx_i + b ≤ −1 (negative side, at most −1).
- So a point is not only on the correct side, it is at least a certain distance from the hyperplane. Positive alone is not enough; it must be ≥ 1.
- Minimizing ‖w‖ makes that "certain distance" as large as possible.

## Margin size
- Points exactly on the boundary of the slab satisfy y_i f(x_i) = 1.
- Distance of such a point from the hyperplane = 1/‖w‖ = **M**.
- Width of the whole slab = **2/‖w‖** = 2M.
- Smaller ‖w‖ → bigger margin.

## Worked example 1 (1D)
Points: x = 1 with y = −1, x = 3 with y = +1.
- Constraints:
  - 3w + b ≥ 1
  - −(1·w + b) ≥ 1, i.e. w + b ≤ −1
- Subtract the two: 2w ≥ 2, so w ≥ 1.
- Minimum norm: **w = 1, b = −2**.
- Boundary: x − 2 = 0 → **x = 2** (the midpoint).
- M = 1/‖w‖ = **1** on each side, total slab width 2. Both points sit exactly at distance 1.

## Worked example 2 (2D)
Points: (0,0) with y = −1, (2,2) with y = +1.
- By symmetry, w points along (1,1): w = c(1,1), and the boundary passes through the midpoint (1,1), so b = −2c.
- Constraint at (2,2): 4c − 2c = 2c ≥ 1 → **c = ½**.
- w = (0.5, 0.5), b = −1. Check (0,0): −1·(0 − 1) = 1 ✓.
- ‖w‖ = √0.5 ≈ 0.707, so M = 1/0.707 ≈ **1.414**.
- Sanity check: the distance between the points is √8 ≈ 2.83, and half of that is 1.414 ✓. The boundary is the perpendicular bisector.

## Perceptron vs max-margin
| | Perceptron | Max-margin (SVM) |
|---|---|---|
| Solution | Any separating hyperplane | One specific optimal hyperplane |
| Depends on start / order | Yes | No |
| Objective | Reduce distance of misclassified points | Maximize distance of closest point (margin) |
| Constraint view | Fix mistakes one by one | y_i f(x_i) ≥ 1 for all i |

## Key takeaways
- Optimal hyperplane = the one with the **largest margin**: the closest point from each class is equally far from it.
- Margin M = distance of the closest point to the hyperplane.
- Start: maximize M subject to y_i(x_iᵀw + b) ≥ M and ‖w‖ = 1.
- Move the norm into the constraint, then set ‖w‖ = 1/M. The constraint becomes y_i(x_iᵀw + b) ≥ 1.
- Maximizing M ⇔ minimizing ‖w‖ ⇔ minimizing ½‖w‖².
- Final problem: **minimize ½‖w‖² subject to y_i(x_iᵀw + b) ≥ 1 for all i**.
- Margin width = 2/‖w‖.
- This assumes **linearly separable** data. The next step is to solve this constrained optimization problem.