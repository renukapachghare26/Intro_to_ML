# Perceptron Learning Algorithm

## Where this fits
- Two ways to build linear classifiers:
  1. **Discriminant functions** (notes 01–05): compare δ_k(x) to get the boundary.
  2. **Model the separating hyperplane directly** (this note and the next).
- Two direct methods are covered:
  - The **perceptron**: leads into neural networks.
  - The **optimal hyperplane** (next class): the most popular way to build classifiers nowadays.
- Notation: the lecture uses β and β0. These notes use **w** and **b**, so β = w and β0 = b.

## Separating hyperplane: fundamentals
- Define f(x) = wᵀx + b.
- The hyperplane L is the set of all x with **f(x) = 0**.
- Sign of f(x) tells the side:
  - f(x) > 0 → one side (class +1)
  - f(x) < 0 → other side (class −1)

### Property 1: w is normal to L
- Take two points x1, x2 on L.
- Then wᵀx1 + b = 0 and wᵀx2 + b = 0.
- Subtract: **wᵀ(x1 − x2) = 0**.
- (x1 − x2) lies along L, so **w is perpendicular to L**.
- Unit normal: **w* = w / ‖w‖**.

### Property 2: any point x0 on L satisfies wᵀx0 = −b

### Property 3: signed distance of a point from L
- Take any point x0 on L. For a point x not on L:

      distance = w*ᵀ (x − x0)

  - Projects (x − x0) onto the normal direction, so it gives the perpendicular distance.
  - **Signed**: the sign depends on which side of L the point lies.
- Substitute w* = w/‖w‖ and wᵀx0 = −b:

      distance = (wᵀx + b) / ‖w‖ = **f(x) / ‖w‖**

- Since f′(x) = w (derivative of f with respect to x):

      distance = f(x) / ‖f′(x)‖

- So f(x) is **proportional** to the signed distance.
- If ‖w‖ = 1, then **f(x) is exactly the signed distance** to the hyperplane.
- Keep this in mind: minimizing/maximizing f(x) means minimizing/maximizing the distance of points from the hyperplane.

## Background (history, brief)
- The perceptron started the **first boom** of artificial neural networks (1950s–60s).
  - People thought it could solve the human learning problem.
  - Example: a perceptron trained to produce speech sounded like a baby learning to speak, and people went crazy over it.
- **Second boom**: the backpropagation algorithm, which fixed some perceptron problems.
- **Third boom**: deep learning.

## The idea
- Want a boundary that classifies the training points correctly.
- For any point that is **misclassified**, keep it as **close to the boundary as possible**.
  - If it ends up on the correct side, great.
  - If it is on the wrong side, at least it is not wrong by a huge margin.
- We use **signed distances**, so being on the right side is "good" and on the wrong side is "bad".

## Setup and misclassification
- Labels: y_i ∈ {+1, −1}.
- Rule:
  - wᵀx_i + b > 0 → predict +1
  - wᵀx_i + b < 0 → predict −1
- **Misclassified** when:
  - true y = +1 but f(x) < 0, **or**
  - true y = −1 but f(x) > 0
- Both cases are captured by one quantity: **y_i · f(x_i)**
  - **Negative** for misclassified points.
  - **Positive** for correctly classified points.

## Perceptron criterion (objective function)
- Let M = the set of misclassified points.

      D(w, b) = − Σ_{i ∈ M} y_i (wᵀx_i + b)

- Each term y_i f(x_i) is negative for misclassified points, so the minus sign makes **D ≥ 0**.
- D is proportional to the total distance of the misclassified points from the hyperplane.
- **Goal**: minimize D. The best case is **M empty** (D = 0, no mistakes).
- As long as M is non-empty, we have not reached the optimum.
- This only works if the data is **linearly separable** (so that M can actually become empty).

## Optimizing: gradient descent → stochastic gradient descent
### Gradient

    ∂D/∂w = − Σ_{i ∈ M} y_i x_i
    ∂D/∂b = − Σ_{i ∈ M} y_i

### Why not just set the gradient to 0 and solve?
- When w changes, the set **M changes** (different points become misclassified).
- So the gradient is only valid at the current (w, b). After moving, it must be **recomputed**.
- We cannot jump a long way using one gradient.
- Hence: take a **step** in the gradient direction, recompute, repeat.

### Gradient descent recap
- Find the direction of steepest ascent and move in the **opposite** direction.
- Step size is proportional to the gradient (scaled by a learning rate ρ).

### Stochastic gradient descent (SGD)
- Do not wait to collect all misclassified points.
- As soon as you find **one** misclassified point, use it as an **estimate of the gradient** and update.
- Hope that on average (in expectation) you move in the right direction.
- This is exactly how the original perceptron algorithm was derived.
- You can also compute the full gradient over all of M and update once per pass. That is perfectly valid too.

## The perceptron update rule
For a misclassified point (x_i, y_i):

    w ← w + ρ · y_i · x_i
    b ← b + ρ · y_i

- Meaning: **multiply the misclassified point by its desired output and add it to the weight vector**.
- Usually **ρ = 1**.

### Algorithm
1. Initialize w and b (e.g. all zeros).
2. Go through the training points.
3. If a point is misclassified (y_i (wᵀx_i + b) ≤ 0), apply the update.
4. Repeat passes until no point is misclassified (M empty).

### About the step size ρ
- In general SGD, ρ must be **very small** (order of 10⁻³ or 10⁻⁴).
  - You are averaging many estimates of the gradient in a local region.
  - A large ρ takes you out of the region after one (possibly wrong) estimate.
- But in the perceptron algorithm, **ρ = 1 works** and convergence can be proven.
- This is how the original algorithm was stated.

## Worked example (2D, ρ = 1)
Points: P1 = (2,3) with y = +1, P2 = (3,1) with y = −1, P3 = (1,1) with y = −1.
Start w = (0,0), b = 0. Misclassified if y · score ≤ 0.

**Pass 1**
- P1: score 0 → mistake. w = (2,3), b = 1.
- P2: score = 6+3+1 = 10, y = −1 → mistake. w = (2,3) − (3,1) = (−1,2), b = 1 − 1 = 0.
- P3: score = −1+2+0 = 1, y = −1 → mistake. w = (−1,2) − (1,1) = (−2,1), b = −1.

**Pass 2**
- P1: score = −4+3−1 = −2, y = +1 → mistake. w = (−2,1) + (2,3) = (0,4), b = 0.
- P2: score = 0+4+0 = 4, y = −1 → mistake. w = (0,4) − (3,1) = (−3,3), b = −1.
- P3: score = −3+3−1 = −1, y = −1 → correct.

**Pass 3**
- P1: score = −6+9−1 = 2 → correct.
- P2: score = −9+3−1 = −7 → correct.
- P3: score = −1 → correct.

No mistakes → **converged**: w = (−3,3), b = −1.
Boundary: −3x1 + 3x2 − 1 = 0, i.e. x2 = x1 + 1/3.
Signed distance of P1 = 2 / ‖w‖ = 2 / 4.24 ≈ 0.47.

## Convergence and problems
### If the data is linearly separable
- Linearly separable = there exists a hyperplane that makes **no mistakes**.
- The perceptron learning algorithm **converges** to some separating hyperplane.

### Problem 1: many solutions, no control
- There are **infinitely many** separating hyperplanes.
- The algorithm converges to **one** of them, and it depends on the starting point (and the order of the data).
- You cannot say which one a priori. You just have to run it.
- Fix (next class): define a single **optimal** hyperplane.

### Problem 2: can be very slow
- Slow especially when the **gap between the classes is small**.
- ρ = 1 makes the weights **oscillate**: a mistake on one side, then back, again and again.
- A small ρ takes only small steps, which is also slow.
- So there is a trade-off in choosing ρ.
- Possible fix: **basis expansion / transform the data** (e.g. x², x³) to widen the gap between classes so it converges faster.

### Problem 3: not linearly separable
- No hyperplane makes M empty.
- So the algorithm **keeps updating forever**, because M never becomes empty.
- It **loops**. The loops can be very large on big datasets.
- **Hard to detect**: you cannot tell whether it is just slow or really looping.
- |M| (number of misclassified points) is **not guaranteed to decrease monotonically**, even when converging, so it does not work as a clean stopping test.
- No efficient general way around this.

### The big drawback: XOR
- Even simple problems like **XOR** are not linearly separable.
- XOR example: (0,0) → −, (1,1) → −, (0,1) → +, (1,0) → +. No single line separates them.
- This (and similar failures) killed the first boom of neural networks: "if it can't solve XOR, forget it".

## Quick comparison with the earlier classifiers
| | Indicator regression | Logistic regression | LDA | Perceptron |
|---|---|---|---|---|
| Approach | Discriminant | Discriminant | Discriminant | Model hyperplane directly |
| Learns | β for each class | β by max likelihood | μ, Σ, π | w, b by fixing mistakes |
| Output | Scores | Probability | Class (posterior) | Class label only |
| Optimizes | Squared error | Likelihood | Fisher / Bayes | Distance of misclassified points |

## Key takeaways
- Separating hyperplane: f(x) = wᵀx + b = 0; **w is normal** to it.
- **Signed distance** to the hyperplane = f(x)/‖w‖ (exact when ‖w‖ = 1).
- Misclassified ⇔ y_i f(x_i) < 0.
- Perceptron criterion: D = −Σ_{i∈M} y_i f(x_i). Minimize it; best case M is empty.
- Gradient changes whenever M changes → use **stochastic gradient descent**: update on one misclassified point at a time.
- Update: **w ← w + y x, b ← b + y** (ρ = 1 works for the perceptron).
- Separable data → converges to **some** hyperplane (which one depends on the start).
- Problems: not unique, slow for small gaps, loops on non-separable data (hard to detect), cannot solve XOR.
- Next: define a single **optimal** separating hyperplane (the margin idea).