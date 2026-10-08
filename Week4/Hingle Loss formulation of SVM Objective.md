# Hinge Loss Formulation of the SVM Objective

## Recap: the primal we had

      minimize   ½ ‖β‖² + C Σ_i ξ_i
      subject to y_i f(x_i) ≥ 1 − ξ_i,   ξ_i ≥ 0

- f(x_i) = x_iᵀβ + β0.
- Here we rewrite the same problem in a different way: as **loss + penalty**, just like ridge regression.

## Step 1: eliminate the slack variables
- For each i the constraints say ξ_i ≥ 0 and ξ_i ≥ 1 − y_i f(x_i).
- We are **minimizing** Σξ_i, so at the optimum each ξ_i takes the **smallest value allowed**:

      ξ_i = max(0, 1 − y_i f(x_i))  =  [ 1 − y_i f(x_i) ]₊

- The **plus notation** [z]₊ means: **count z only when it is positive; read it as 0 when it is negative.**
- So ξ_i is exactly the hinge loss of point i. No separate constraints are needed any more.

## Step 2: the new objective
- Substitute and divide everything by C. Set **λ = 1/C** (the lecture writes it with a λ replacing the α-style multiplier):

      minimize   Σ_{i=1}^{N} [ 1 − y_i f(x_i) ]₊  +  (λ/2) ‖β‖²

- Check: ½‖β‖² + CΣξ_i divided by C gives (1/C)·½‖β‖² + Σξ_i, which is the same thing with λ = 1/C.
- Large C ⇔ small λ (weak penalty on the norm, fit the data hard). Small C ⇔ large λ (strong penalty on the norm, wider margin).

## Hinge loss: shape
- Loss of one point as a function of z = y_i f(x_i):

      L(z) = [ 1 − z ]₊ = 0            if z ≥ 1
                         = 1 − z       if z < 1

- For z ≥ 1: loss is **0** (correct side and at least a margin away).
- For z < 1: loss grows **linearly** as z decreases (inside the margin, then wrong side).
- The picture looks like a **door or a book opening on a hinge**: two flaps meeting at z = 1. That is why it is called **hinge loss**.
- You may have read elsewhere that "**SVMs minimize hinge loss**". This is exactly what that means.

## Link to ridge regression
- Ridge regression: **loss (squared error) + penalty (L2 norm of β)**.
- SVM in this form: **loss (hinge) + penalty (L2 norm of β)**.
- Same structure. Before, we treated ‖β‖² as the objective and the correctness conditions as constraints, then moved constraints into the objective via the Lagrangian. Now we write it directly as loss + penalty.
- So SVM also uses an **L2 penalty** on β, as in ridge.

## What each part means
- **Constraints (now the hinge loss)**: enforce **correctness**. Points should be on the right side, and a certain distance away.
- **‖β‖² term**: enforces **robustness** (a small-norm solution = large margin = far from the hyperplane).
- The constraint is an essential part of the problem: it is not only the distance from the hyperplane that matters, but also being on the **correct side**.
- Writing it as a hinge loss makes this explicit: "this is the loss I care about" (correctness), plus a penalty (small norm).

## What if we change the penalty? (L1)
- Replace ‖β‖² with ‖β‖₁ (lasso-style penalty) → **L1-regularized SVM**.
- It is a valid thing to do, but a **harder optimization problem**.
- Expected effect (as in lasso): it tends to make **many coefficients exactly 0** (sparsity).
- The lecture raised the question of whether this would reduce the number of **support vectors**, and left it as a thinking exercise. The sparsity from the L1 penalty acts on the coefficients of β.

## Other loss functions on the same footing
We can plug other losses into "loss + penalty". Write z = y f(x), with y ∈ {+1, −1}.

| Loss | Formula | Behaviour |
|---|---|---|
| **0–1 loss** | 1 if z ≤ 0 (wrong), 0 if z > 0 | What we really want |
| **Hinge** | [1 − z]₊ | 0 for z ≥ 1, linear below |
| **Squared** | (1 − z)² | Penalizes even when z > 1 |
| **Logistic** | ln(1 + e^(−z)) | Smooth, never reaches 0 |

### Squared loss: why it is not ideal
- Usually written (y − f(x))². Since y = ±1, y − f = y(1 − y f), so **(y − f)² = (1 − y f)²**. Same thing.
- Problem: a point that is on the **correct** side and **far** from the hyperplane (z large) still has loss, because the error is squared in both directions.
- You are penalized for being "too correct". That is weird.
- Hinge loss is 0 there. **Hinge loss more often than not gives a better solution than squared error.**

### 0–1 loss: what we really want
- 0 if the point is correctly classified, 1 if not.
- Is the smallest achievable 0–1 loss equal to 0? It depends on:
  1. the **data**, and
  2. the **family of classifiers** you chose (e.g. linear). If the family cannot separate the data, the minimum is above 0.
- "Minimize the 0–1 loss" means: the best achievable error given the data distribution and the classifier family.
- A line of theory research asks: **if I minimize a different (surrogate) loss such as hinge or squared, do I end up near the solution of the 0–1 loss?** (Just for interest; not tested.)

### Logistic loss
- Logistic regression (maximum likelihood) actually minimizes this loss, even though we never wrote it as a loss.
- For labels y ∈ {+1, −1}: **ln(1 + e^(−y f(x)))**. It is the negative log-likelihood from note 02 in different notation.
- It never reaches exactly 0 (it decays smoothly), but you can still minimize it. (Covered again later.)

## Loss values at a glance

| z = y·f(x) | 0–1 | Hinge | Squared | Logistic |
|---|---|---|---|---|
| −2 | 1 | 3 | 9 | 2.13 |
| −1 | 1 | 2 | 4 | 1.31 |
| 0 | 1 | 1 | 1 | 0.69 |
| 0.5 | 0 | 0.5 | 0.25 | 0.47 |
| 1 | 0 | 0 | 0 | 0.31 |
| 2 | 0 | 0 | **1** | 0.13 |
| 3 | 0 | 0 | **4** | 0.05 |

- Note the squared loss **rising again** for z > 1 (4 at z = 3), while hinge stays at 0.
- Hinge and logistic both grow roughly linearly for very negative z, so they are less dominated by outliers than the squared loss.

## Worked example (continuing the soft-margin note)
Same points and hyperplanes as in `SVMs_for_Linearly_Non_Separable_Data.md`:
- Points (x, y): (5, +1), (3, +1), (2.4, +1), (1.5, +1), (0.5, −1).
- **Hyperplane A**: f(x) = x − 2, ‖β‖² = 1.
  - z = y·f: 3, 1, 0.4, −0.5, 1.5.
  - Hinge losses: 0, 0, 0.6, 1.5, 0. These are exactly the slacks ξ_i found earlier. Total = **2.1**.
  - Objective = 2.1 + (λ/2)(1) = **2.1 + 0.5λ**.
- **Hyperplane B**: f(x) = 2x − 3, ‖β‖² = 4.
  - Only x = 1.5 has z = 0 → hinge loss 1. Total = **1**.
  - Objective = 1 + (λ/2)(4) = **1 + 2λ**.
- A is better when 2.1 + 0.5λ < 1 + 2λ → 1.1 < 1.5λ → **λ > 0.73**, i.e. **C = 1/λ < 1.36**.
- This matches the crossover found earlier in the C-formulation. It is the same problem written two ways.

## Key takeaways
- At the optimum ξ_i = [1 − y_i f(x_i)]₊, so the SVM can be written without constraints:
  **minimize Σ [1 − y_i f(x_i)]₊ + (λ/2)‖β‖²**, with λ = 1/C.
- [z]₊ means "count z only when positive".
- **Hinge loss** is 0 once a point is at least a margin on the correct side, and linear otherwise (like a book on a hinge).
- It has the same form as **ridge regression**: loss + L2 penalty.
- The hinge loss carries the **correctness** requirement; the norm penalty carries **robustness** (margin).
- An L1 penalty (lasso-style) is possible but harder, and promotes sparse coefficients.
- Squared loss penalizes points that are correct and far away, so hinge is usually better.
- The loss we truly want is 0–1; hinge, squared and logistic are surrogates. Logistic regression minimizes the logistic loss ln(1 + e^(−y f)).