# Support Vector Machines II: Interpretation and Analysis

## Where we are
- From note 07, the optimal (max-margin) hyperplane problem is:

      minimize   ½ ‖β‖²
      subject to y_i (x_iᵀβ + β0) ≥ 1,   i = 1..N

- (Lecture notation β, β0 = w, b in earlier notes.)
- This is a **simple optimization problem**: a **quadratic objective** with **linear constraints** (a convex problem).
- Standard route (from the convex optimization tutorial): write the **Lagrangian**, then go to the **dual**.

## Step 1: the Lagrangian (primal)
- Apply the constraint for every data point, i = 1 to N, each with a multiplier α_i ≥ 0:

      L_P = ½ ‖β‖² − Σ_{i=1}^{N} α_i [ y_i (x_iᵀβ + β0) − 1 ]

- The "P" stands for **primal**.
- We form the **dual** because it is a lot easier to solve.

## Step 2: derivatives
- Differentiate L_P and set to 0:
  - With respect to β:  **β = Σ α_i y_i x_i**
  - With respect to β0:  **Σ α_i y_i = 0**
- Substitute these back into L_P and simplify (the lecture skips this algebra; it is worth working through once).
- Because of the β² term, you get **α_i α_j y_i y_j** cross terms.

## Step 3: the dual problem

      maximize   Σ_i α_i  −  ½ Σ_i Σ_j α_i α_j y_i y_j x_iᵀx_j
      subject to α_i ≥ 0,   Σ_i α_i y_i = 0

- Constraints are now much simpler: mainly **α_i ≥ 0**.
- Efficient algorithms exist for this form, and many packages solve SVMs for you.
- Still, you should know **what optimization problem is being solved**. Do not use it as a black box.
- **Duality gap = 0** here (primal and dual have the same optimal value), so solving the dual solves the primal.

## KKT conditions
At the solution, all KKT conditions must hold:
1. **Primal feasible**: y_i (x_iᵀβ + β0) ≥ 1 for all i (a valid solution of the primal).
2. **Dual feasible**: α_i ≥ 0.
3. **Stationarity**: β = Σ α_i y_i x_i and Σ α_i y_i = 0.
4. **Complementary slackness**:

       α_i [ y_i (x_iᵀβ + β0) − 1 ] = 0   for every i

   - (In the optimization notes this appears as λ_i f_i(x) = 0; here it is α_i × the constraint.)
- These conditions tell us what the solution looks like (next sections).

## Insight 1: form of β
- **β = Σ α_i y_i x_i**
- β is built by **picking certain training points, multiplying each by its label (+1 / −1) and a weight α_i, and adding them up**.
- Label sign: a positive point is added, a negative point is subtracted.
- **Link to the perceptron**:
  - Perceptron: keep adding misclassified points (times label) to the weight vector.
  - SVM: also a sum of data points times labels, but derived properly from "maximize the margin" instead of a heuristic.
  - The perceptron used gradient descent on an arbitrarily chosen set of misclassified points.
  - Here we started from minimizing distance to the closest point and **derived** a perceptron-like form.
- These days, when people say "train a perceptron" they more often mean this approach than the original perceptron learning rule.

## Insight 2: support vectors (from complementary slackness)
- Condition: α_i [ y_i (x_iᵀβ + β0) − 1 ] = 0.
- The product is 0 when **either** α_i = 0 **or** the bracket = 0.
- Geometric meaning:
  - **Point exactly on the margin** (the closest points): y_i f(x_i) = 1, so the bracket is 0 → α_i may be non-zero.
  - **Point farther from the hyperplane**: y_i f(x_i) > 1, so the bracket is non-zero → **α_i must be 0**.
- Therefore:
  - Points far from the hyperplane **do not contribute** to β (their α_i = 0).
  - Only points **on the margin** contribute.
- Such margin points are called **support points** or **support vectors**.
- **β depends only on the support vectors.**
- Example in the lecture's picture: only two points lie on the margin, so only two support vectors.

### Finding β0
- Take any support vector and plug it in: y_i (x_iᵀβ + β0) = 1. Solve for β0.
- In theory every support vector gives the same β0. In practice they differ slightly for **numerical reasons**.
- Practice: compute β0 from **each support vector** and **take the average**.
- Then the hyperplane is f(x) = x ᵀβ + β0.

### Degenerate cases
- α_i = 0 even though the point is on the margin can happen if data is degenerate, e.g. two data points at exactly the same position (repeated points).
- These are special cases; the usual picture holds in general.

## Insight 3: only dot products appear
- In the dual: only **x_iᵀx_j** appears.
- In the final classifier: f(x) = Σ α_i y_i x_iᵀx + β0, so only x_iᵀx appears.
- If there is a very efficient way to compute dot products xᵀx′, we can do clever things (to be revisited later). Just remember this observation.

## Worked example (continuing note 07, 1D)
Points: x = 1 with y = −1, x = 3 with y = +1.

**Dual:**
- Σ α_i y_i = 0 → α_1 = α_2 = α (both multipliers equal).
- β = Σ α_i y_i x_i = α(3) − α(1) = **2α**.
- Double sum: i=j=1 gives 1, i=j=2 gives 9, the two cross terms give −3 each → total 4.
- Dual objective = 2α − ½(α² · 4) = **2α − 2α²**.
- Maximize: derivative 2 − 4α = 0 → **α = ½**.
- So β = 2α = **1** ✓ (same as the direct solution in note 07).

**β0:**
- Both points are support vectors. Use x = 3 (y = +1): 3β + β0 = 1 → **β0 = −2** ✓.
- Boundary x = 2, margin 1 on each side.

**Adding a far point:** add x = 5 with y = +1.
- Constraint value: 1 · (5 − 2) = 3 > 1 → the bracket is 2 ≠ 0 → **α = 0** for this point.
- β and β0 do not change, so the boundary stays at x = 2. A far point has no influence.

**2D check:** points (0,0) with y = −1 and (2,2) with y = +1.
- β = α(2,2) − 0 = 2α(1,1). Only the (2,2) term survives in the double sum: 8α².
- Dual = 2α − 4α² → α = ¼ → β = (0.5, 0.5).
- From (2,2): 0.5·2 + 0.5·2 + β0 = 1 → β0 = −1. Matches note 07.

## Comparison with LDA
### How LDA depends on the data
- LDA estimates a **density** (assumes Gaussian classes, equal covariance).
- Every training point contributes to the estimated μ, Σ and π.
- So **every** point (near or far from the boundary) affects the class boundary.
- Hence LDA is **more susceptible to noise**: even a noisy point far from the boundary shifts the estimates.

### How the optimal hyperplane depends on the data
- Only points **near the boundary** (support vectors) matter.
- Points can be moved around elsewhere with no effect.
- Only noise that lands close to the decision surface changes the classifier.
- If the noise is spread uniformly, LDA is affected more. The optimal hyperplane is affected only by the fraction of noise that changes the actual decision surface.

### The trade-off
- If the data is **truly Gaussian with equal covariance**, **LDA is optimal** (the best possible classifier).
- The optimal hyperplane approach depends on the actual data you get.
- But in general it is **more preferable because it is more stable**.

### Stability
- **Stable** = small changes in the data do not change the classifier much.
- Moving a support vector changes the boundary.
- Moving non-support points around changes nothing, as long as the **set of support vectors stays the same** (e.g. do not move a point closer to the hyperplane than the existing support vectors).
- Same support vectors → same classification surface again and again.
- This is why the max-margin approach (SVM) is very stable. (SVMs come up properly in the next topic.)

## Summary table
| | LDA | Optimal hyperplane (SVM) |
|---|---|---|
| Uses | Estimated density (μ, Σ, π) | Margin points only |
| Points that matter | All of them | Support vectors only |
| Noise sensitivity | Higher | Lower (only noise near the boundary) |
| Optimal when | Data truly Gaussian, equal covariance | No such assumption needed |
| Stability | Lower | Higher |

## Key takeaways
- Primal: minimize ½‖β‖² subject to y_i(x_iᵀβ + β0) ≥ 1. A quadratic objective with linear constraints.
- Form the Lagrangian with α_i ≥ 0, differentiate, substitute, and you get the **dual** (easier to solve). Duality gap is 0.
- Stationarity gives **β = Σ α_i y_i x_i** and Σ α_i y_i = 0.
- Complementary slackness: α_i [y_i f(x_i) − 1] = 0.
  - Points on the margin → α_i can be non-zero (**support vectors**).
  - Points away from the margin → α_i = 0.
- β depends **only on the support vectors**.
- Get β0 from the support vectors and **average** the values.
- The dual and the classifier use only dot products x_iᵀx_j (useful later).
- Compared with LDA, the optimal hyperplane is **more stable and less noise-sensitive**, though LDA is optimal if the data is truly Gaussian with equal covariance.