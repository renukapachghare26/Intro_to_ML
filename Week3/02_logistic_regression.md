# 02 – Logistic Regression

## Why do we need it?
- Linear regression on 0/1 indicators gives outputs that are **not bounded** to [0, 1], so they are not real probabilities.
- Ideal goal: f(x) = P(y | x). Modelling that directly with a line is hard.
- Fix: do not fit the probability itself. Fit a **transformation of the probability** with a linear model.
- The transformation used is the **logit (log-odds)**.

## Idea
Linear classification gives a raw score z = w · x + b that can be any number.
Logistic regression turns that score into a **probability between 0 and 1** using the **sigmoid function**:

    σ(z) = 1 / (1 + e^(-z))

    P(y = 1 | x) = σ(w · x + b)

Despite the name, it is a **classification** method.

- Setup for now: **binary** classification, labels 0 or 1.
- p(x) = P(y = 1 | x), so 1 − p(x) = P(y = 0 | x).
- The linear part is unbounded, but the sigmoid squeezes it into (0, 1).
- Decision rule: p(x) > 0.5 → output 1, else output 0.
- Where the 0.5 point falls depends on the bias (β0).

## Sigmoid values (feel for it)
| z | σ(z) |
|---|---|
| -4 | 0.018 |
| -2 | 0.119 |
| 0 | **0.5** |
| 2 | 0.881 |
| 4 | 0.982 |

- z = 0 gives probability 0.5, which is the decision boundary.
- Large positive z gives probability close to 1. Large negative z gives probability close to 0.

## Odds and log-odds (why it is "linear")
- Odds = p / (1 - p) = probability of success / probability of failure.
- Log-odds (**logit**) = ln( p / (1 - p) ) = **w · x + b**

So the **log-odds are a linear function of x**. This is why the boundary is still linear.

Example: p = 0.8 → odds = 0.8/0.2 = 4 ("4 to 1") → log-odds = ln 4 ≈ 1.386.

## Decision boundary is still a hyperplane
- Boundary: p(x) = 0.5.
- Then p/(1 − p) = 1, so log-odds = ln 1 = 0.
- So the boundary is **β0 + βᵀx = 0**, which is a straight line / hyperplane.
- Looks complicated (exponentials) but the separating surface is linear.

## Worked example
Hours studied → pass probability. Suppose we learned w = 1.2, b = -4.8.

    z = 1.2 * hours - 4.8

| Hours | z | P(pass) = σ(z) | Prediction (threshold 0.5) |
|---|---|---|---|
| 2 | -2.4 | 0.083 | Fail |
| 4 | 0.0 | 0.500 | Boundary |
| 6 | 2.4 | 0.917 | Pass |

Boundary: 1.2*h - 4.8 = 0 → h = **4 hours**.
Interpretation of w: each extra hour multiplies the odds of passing by e^1.2 ≈ **3.3**.

## Why it is popular in practice
- Looks simple but gives a **very powerful classifier** that works well in practice.
- Used widely by both ML people and statisticians.
- Used for **sensitivity analysis**: look at the β vector to see how much each factor contributes to the predicted class.
- Often the preferred (sometimes the only accepted) classifier in fields like medicine, because it is interpretable and works well.

## Multi-class logistic regression
- Each class gets its own expression with its own β0 and β.
- Only **K − 1 sets of coefficients** are needed for K classes. The K-th class probability is determined automatically (probabilities sum to 1).
- Convention: pick one class (first or last, arbitrary) as the **reference class** and set its coefficients to 0.

      P(G = k | x) = exp(β_k0 + β_kᵀx) / (1 + Σ_{l=1}^{K-1} exp(β_l0 + β_lᵀx)),   k = 1..K-1
      P(G = K | x) = 1 / (1 + Σ_{l=1}^{K-1} exp(β_l0 + β_lᵀx))

- For each class the numerator has only that class's terms; the denominator has all of them.
- Alternative view (below): **softmax**.

### Softmax
For K classes: P(class k | x) = e^(z_k) / Σ_j e^(z_j)

Example: scores z = (2, 1, 0) → exp = (7.39, 2.72, 1) → sum = 11.11 → probabilities = (0.665, 0.245, 0.090).

## How are the parameters estimated? Maximum likelihood
- So far we minimized an **error function** (e.g. squared error in linear regression).
- Here we **maximize the likelihood of the data** instead.
- Training data D is **fixed**. Likelihood is a function of the parameters θ:

      L(θ) = P(D | θ)

- In logistic regression, θ = the betas.
- We usually work with the **log-likelihood** l(θ) (log turns products into sums and simplifies the distributions).
- (Maximum likelihood is covered in more detail in a later session.)

### Likelihood for the two-class case
- Data = pairs (x_i, g_i), with g_i ∈ {0, 1}.
- Probability of one pair: p(x_i)^g_i · (1 − p(x_i))^(1 − g_i)
  - If g_i = 1, only the p(x_i) term remains.
  - If g_i = 0, only the (1 − p(x_i)) term remains.
- Assuming points are sampled **independently**, multiply over all points.
- Log-likelihood:

      l(β) = Σ [ g_i ln p(x_i) + (1 − g_i) ln(1 − p(x_i)) ]
           = Σ [ g_i (β0 + βᵀx_i) − ln(1 + e^(β0 + βᵀx_i)) ]

### Gradient and why it is not closed-form
- Set the derivative of l with respect to each β_j to 0:

      ∂l/∂β_j = Σ x_ij (g_i − p(x_i))

- Looks easy but p(x_i) contains an exponential in β, so there is **no closed-form solution**.
- Need an **iterative method**. Most popular: **Newton-Raphson**.

## Loss function: cross-entropy (log loss)
Same thing as the negative log-likelihood. For true label y in {0, 1} and predicted p:

    Loss = -[ y * ln(p) + (1 - y) * ln(1 - p) ]

Intuition:
- True y = 1, predicted p = 0.9 → loss = -ln(0.9) = **0.105** (small, good)
- True y = 1, predicted p = 0.1 → loss = -ln(0.1) = **2.303** (big, bad)
Confident wrong predictions are punished heavily.

Total loss over N points = average of the individual losses.
Minimizing it is the same as **maximum likelihood estimation**.

## Training option 1: gradient descent
Idea: start at a current solution, compute the gradient, take a **small step opposite to it** (to minimize loss). Repeat.

The gradients are neat and simple:

    ∂Loss/∂w = (1/N) * Σ (p_i - y_i) * x_i
    ∂Loss/∂b = (1/N) * Σ (p_i - y_i)

Update:

    w = w - α * ∂Loss/∂w
    b = b - α * ∂Loss/∂b

(α = learning rate)

### One tiny step by hand
One data point: x = 2 hours, y = 1 (passed). Start w = 0, b = 0, α = 0.1.
1. z = 0 → p = 0.5
2. Error = p - y = 0.5 - 1 = -0.5
3. ∂w = -0.5 * 2 = -1, ∂b = -0.5
4. w = 0 - 0.1*(-1) = **0.1**, b = 0 - 0.1*(-0.5) = **0.05**
The weights moved so the pass probability for 2 hours goes up.

## Training option 2: Newton-Raphson and IRLS
### Newton-Raphson idea
- Take the old estimate and adjust it using **first derivative / second derivative**:

      β_new = β_old − (l′ / l″)

- Uses curvature (second derivative), so it takes smarter steps than plain gradient descent.

### Matrix notation
- **X**: N × (p+1) input matrix (with the column of 1s).
- **p**: N-vector, entry i = probability that x_i is class 1 (under current β).
- **g** (or y): N-vector of 0/1 labels.
- **W**: N × N **diagonal** matrix, entry i = p_i (1 − p_i).
- First derivative: ∂l/∂β = **Xᵀ(g − p)**
- Second derivative: ∂²l/∂β∂βᵀ = **−XᵀWX**

### Newton update

    β_new = β_old + (XᵀWX)⁻¹ Xᵀ (g − p)

- Start from any guess; **all betas = 0 works fine**.
- Rearranging (a bit of algebra: write β_old as (XᵀWX)⁻¹XᵀWXβ_old, so W⁻¹ and W cancel):

      β_new = (XᵀWX)⁻¹ XᵀW z

  where the **adjusted response** is

      z = Xβ_old + W⁻¹ (g − p)

- Xβ_old = the prediction with old parameters; W⁻¹(g − p) = the adjustment to it.
- This looks like linear regression: (XᵀX)⁻¹Xᵀy becomes (XᵀWX)⁻¹XᵀWz.

### Weighted least squares
- Ordinary least squares minimizes Σ (error)².
- **Weighted** least squares gives each point its own weight in the squared error.
  - Higher weight → try harder to fit that point.
  - Lower weight → care less about that point.
- Minimizer: β = (XᵀWX)⁻¹ XᵀW z (take derivative, set to 0).
- So each Newton step = solving a **weighted least squares** problem with the adjusted response.

### IRLS (Iteratively Reweighted Least Squares)
Same as Newton-Raphson, described as an algorithm:
1. Start with a guess for β (e.g. all 0).
2. Compute p from β.
3. Build W (diagonal, p(1 − p)).
4. Form the adjusted response z.
5. Solve the weighted least squares problem → new β.
6. Repeat until predictions are accurate enough (β stops changing).

- IRLS is the base logistic regression solver in packages like R.
- Many more efficient solvers exist, but this shows how hard optimization can get.

## Regularization
Add a penalty to prevent overfitting (especially with perfectly separable data, where weights grow forever):
- L2: Loss + λ * ||w||²
- L1: Loss + λ * ||w||₁ (can zero out features)

## Evaluation quick list
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC-AUC

## Key takeaways
- P(y=1|x) = σ(w · x + b), which fixes the "outputs outside [0, 1]" problem of linear regression.
- Log-odds (logit) are linear in x, so the boundary (p = 0.5) is still a hyperplane.
- Multi-class needs K − 1 coefficient sets, with one reference class set to 0.
- Trained by **maximum likelihood** = minimizing cross-entropy.
- No closed-form solution → iterate: gradient descent or Newton-Raphson (IRLS).
- IRLS = repeated weighted least squares on an adjusted response.
- The β vector is useful for sensitivity analysis (importance of each feature).
- Output is a probability; the threshold (0.5 by default) is adjustable.