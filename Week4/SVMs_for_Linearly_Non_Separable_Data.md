# SVMs for Linearly Non-Separable Data (Soft Margin)

## The problem
- Notes 07 and 08 assumed the data is **linearly separable**.
- If it is not, the perceptron does **not converge**, and the hard-margin problem (y_i f(x_i) ≥ 1 for all i) has **no solution**.
- Terminology: say "**linearly inseparable**" data. Be careful where the negation goes ("not linearly separable" can be misread).
- Goal: tweak the objective so we can handle inseparable data.
- Many choices are possible. One choice gives a very nice optimization formulation, so it is the one used.

## The idea
- Still want to **maximize the margin**.
- Also want to get **as many points correct as possible**.
- Some points will be **inside the margin**, and some will be on the **wrong side**.
- Trying to classify those correctly could force a boundary that **shrinks the margin a lot**, so it is okay to get a few wrong.
- Margin side matters: for each class the margin is on that class's side of the hyperplane. A point is "inside the margin" if it falls on the hyperplane-facing side of its own class's margin line.
- Condition we want: **y_i f(x_i) ≥ 1** (this was the hard-margin constraint).
  - For a point inside the margin or misclassified, y_i f(x_i) is **less than 1** (and negative if misclassified).
- Plan: **measure how far each point violates the margin** and **minimize these violations** together with the original objective.

## Slack variables ξ_i
- Introduce a **slack variable ξ_i ≥ 0** for every point.
- Original constraint in the geometric form: y_i f(x_i) ≥ M.
- Relaxed: **y_i f(x_i) ≥ M (1 − ξ_i)**.
- Meaning: the point need not be at distance M. It may be a **fraction** of M closer.
- Interpretation of ξ_i = **relative (fractional) distance by which point i violates the margin**.
  - ξ_i = 0 → point is on the margin or on the correct side beyond it (no violation).
  - 0 < ξ_i ≤ 1 → inside the margin but still on the correct side of the hyperplane.
  - ξ_i > 1 → on the **wrong side** of the hyperplane (misclassified).
- If we force **all ξ_i = 0**, we are back to the hard-margin (separable) problem.
- We allow leeway so that **most ξ_i are 0** but a few can be positive.
- This is a **standard technique for relaxing constraints** in optimization.

### Why M(1 − ξ_i) and not M − ξ_i?
- M − ξ_i is also possible (and is a bit more common in other treatments).
- But in this setup it leads to a **non-convex** optimization problem. We do not want that.
- M(1 − ξ_i) keeps the problem convex.

## Constraints on the slacks
1. **ξ_i ≥ 0 for all i**
   - We only care about points going to the **wrong side of the margin**.
   - A negative ξ_i would make the right-hand side **larger than M**, i.e. demand the point be **even farther away** than M. That is a tighter constraint, which is not what we want.
   - ξ_i is a **relative** distance: the requirement becomes M − M·ξ_i.
2. **Σ ξ_i ≤ constant**
   - A **budget** on the total violation. We do not want the violations to be very large in total.

### Why minimize the sum and not the maximum?
- Minimizing the **sum** of deviations lets the total violation be concentrated on a few hard points.
- Example: data is perfectly separable except for one **outlier** on the wrong side.
  - Sum: the outlier can take a big share of the violation without wrecking the rest.
  - Max: you try to keep the largest violation small, so the hyperplane gets **dragged toward the outlier**.
- Other formulations exist, but the sum gives a nice computation (a main reason it is popular).

## From budget to penalty (same trick as ridge / lasso)
- A budget constraint of the form "sum ≤ constant" appeared before in **ridge regression and lasso**.
- There we **pushed the constraint into the objective** with a multiplier.
- There is a relationship between the constant and the multiplier (they are not the same number, but equivalent ways of writing the problem; it also depends on the range of the objective).
- Do the same here:
  - Normalize w using ‖w‖ = 1/M (as in note 07). The M disappears, since M = 1/‖w‖.
  - The constraint becomes y_i f(x_i) ≥ 1 − ξ_i.

## Final primal problem (soft-margin SVM)

      minimize   ½ ‖w‖² + C Σ_{i=1}^{N} ξ_i
      over       w, b, ξ
      subject to y_i (x_iᵀw + b) ≥ 1 − ξ_i,   ξ_i ≥ 0,   i = 1..N

- (Lecture uses β, β0 for w, b.)
- The constraint Σξ_i ≤ constant is **gone** (moved into the objective as C Σξ_i).
- ξ_i ≥ 0 is a condition for **each** i.

## Role of C
- C = how much you **penalize margin violations**.
- **Large C**:
  - Violations are expensive, so ξ_i are forced small.
  - Fits more of the training data correctly.
  - **Smaller margin**.
  - **C → ∞** forces all ξ_i = 0 → the **linearly separable (hard-margin)** case.
- **Small C**:
  - Violations are cheap, so many errors are allowed.
  - **Larger margin**.
- It is a **trade-off between margin width and training errors**.

### Why a smaller C can help even for separable data
- Data may look perfectly separable, but with a few **noisy points near the margin**.
- Hard margin (very large C): you pay attention to those noisy points and end up with a **small margin**.
- Small C: you ignore a few noisy points, make a few **training errors**, and get a **more robust** classifier.
- The resulting hyperplane is also usually **more correct in an expected sense** (on new data).
- So allowing errors can be the better choice when the data is noisy.

## The dual and the KKT conditions
### Lagrangian (primal), with multipliers α_i ≥ 0 and μ_i ≥ 0

      L_P = ½‖w‖² + C Σ ξ_i − Σ α_i [ y_i (x_iᵀw + b) − (1 − ξ_i) ] − Σ μ_i ξ_i

### Derivatives set to 0
- ∂/∂w: **w = Σ α_i y_i x_i**
- ∂/∂b: **Σ α_i y_i = 0**
- ∂/∂ξ_i: **α_i = C − μ_i**

### Dual problem

      maximize   Σ_i α_i − ½ Σ_i Σ_j α_i α_j y_i y_j x_iᵀx_j
      subject to 0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0

- **Same dual as the separable case**, but the constraint changes:
  - Before: only α_i ≥ 0.
  - Now: **0 ≤ α_i ≤ C**.
- Why the upper bound: α_i = C − μ_i and μ_i ≥ 0, so α_i cannot exceed C.

### KKT conditions (7 in the slide)
- Feasibility: y_i f(x_i) − (1 − ξ_i) ≥ 0, ξ_i ≥ 0, α_i ≥ 0, μ_i ≥ 0.
- Stationarity: the three derivative conditions above.
- Complementary slackness:
  - **α_i [ y_i f(x_i) − (1 − ξ_i) ] = 0**
  - **μ_i ξ_i = 0**

## Three cases for each point
- As before, **w = Σ α_i y_i x_i**: only points with α_i ≠ 0 contribute.

| Case | Position | ξ_i | α_i | Support vector? |
|---|---|---|---|---|
| 1 | Correct side, **beyond** the margin (y_i f(x_i) > 1) | 0 | **0** | No |
| 2 | **Exactly on** the margin (y_i f(x_i) = 1) | 0 | 0 ≤ α_i ≤ C | Yes (if α_i > 0) |
| 3 | **Inside** the margin or on the **wrong side** (y_i f(x_i) < 1) | > 0 | **C** | Yes |

### Reasoning for each case
1. **Case 1**: ξ_i = 0, and the bracket y_i f(x_i) − 1 is non-zero (the point is far enough away). Complementary slackness forces **α_i = 0**. Same as the separable case.
2. **Case 2**: bracket = 0 with ξ_i = 0, so α_i is free (between 0 and C).
3. **Case 3**: if ξ_i > 0, then μ_i ξ_i = 0 forces **μ_i = 0**, so **α_i = C − μ_i = C**. Then ξ_i is adjusted so the bracket becomes 0, i.e. **y_i f(x_i) = 1 − ξ_i**.
   - Notice ξ_i = 0 in both cases 1 and 2, because "≥ 1" is satisfied exactly or with room to spare.

### Support vectors now
- **Everything on the margin** and **everything on the wrong side of the margin** (inside it or misclassified).
- All points with α_i ≠ 0.
- Compared with the separable case, there can be **many more** support vectors.
- b is still found by averaging over support vectors on the margin (those with ξ_i = 0, 0 < α_i < C).

## Worked example (computing slacks)
Take a candidate hyperplane **A**: f(x) = x − 2 (w = 1, b = −2). Margin M = 1/‖w‖ = 1. Slack: ξ_i = max(0, 1 − y_i f(x_i)).

| x | y | f(x) | y·f(x) | ξ | Meaning |
|---|---|---|---|---|---|
| 5 | +1 | 3 | 3 | 0 | Far on the correct side |
| 3 | +1 | 1 | 1 | 0 | Exactly on the margin |
| 2.4 | +1 | 0.4 | 0.4 | **0.6** | Inside the margin, still correct |
| 1.5 | +1 | −0.5 | −0.5 | **1.5** | Wrong side (ξ > 1) |
| 0.5 | −1 | −1.5 | 1.5 | 0 | Far on the correct side |

- Σξ = 0.6 + 1.5 = **2.1**
- Objective A = ½‖w‖² + C Σξ = **0.5 + 2.1 C**

Compare candidate hyperplane **B**: f(x) = 2x − 3 (w = 2, b = −3). Margin = 1/2 = 0.5.
- Slacks: x = 1.5 → f = 0, y·f = 0 → ξ = **1**. All other points have y·f ≥ 1 → ξ = 0.
- Σξ = 1
- Objective B = ½(4) + C(1) = **2 + C**

Which is better?
- A < B when 0.5 + 2.1C < 2 + C → 1.1C < 1.5 → **C < 1.36**.
- **Small C** → prefers A (wider margin, more violation).
- **Large C** → prefers B (narrower margin, fewer violations).
- This is the margin vs. errors trade-off in numbers. (A and B are just two candidates, not the true optimum.)

## Why bother understanding this?
- You will use a package (e.g. LibSVM) to solve it, but you should know **what is being solved**.
- Many people run SVMs with **default parameters** and never tune them.
- Knowing what **large C vs small C** does lets you choose C sensibly instead of blindly trying a range of numbers.
- Understanding the internals helps you use the tools better.

## Key takeaways
- Hard-margin SVM fails for linearly inseparable data. Fix: **slack variables ξ_i ≥ 0**.
- Relaxed constraint: y_i f(x_i) ≥ M(1 − ξ_i) (not M − ξ_i, which is non-convex).
- ξ_i = fractional violation of the margin: 0 → fine, 0 to 1 → inside the margin, above 1 → misclassified.
- Budget Σξ_i ≤ const becomes a penalty in the objective, like ridge/lasso.
- **Primal**: minimize ½‖w‖² + C Σξ_i subject to y_i f(x_i) ≥ 1 − ξ_i, ξ_i ≥ 0.
- **Dual**: same as before, but **0 ≤ α_i ≤ C**.
- **Large C** → fewer training errors, smaller margin (C = ∞ is the hard margin). **Small C** → more errors allowed, wider margin, often more robust on noisy data.
- Points beyond the margin: α = 0. Points on the margin: 0 ≤ α ≤ C. Points inside the margin or misclassified: α = C. **Support vectors are every point with α ≠ 0.**
- w = Σ α_i y_i x_i still holds.