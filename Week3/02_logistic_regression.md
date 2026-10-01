# 02 – Logistic Regression

## Idea
Linear classification gives a raw score z = w · x + b that can be any number.
Logistic regression turns that score into a **probability between 0 and 1** using the **sigmoid function**:

    σ(z) = 1 / (1 + e^(-z))

    P(y = 1 | x) = σ(w · x + b)

Despite the name, it is a **classification** method.

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
- Odds = p / (1 - p)
- Log-odds (logit) = ln( p / (1 - p) ) = **w · x + b**

So the **log-odds are a linear function of x**. This is why the boundary is still linear.

Example: p = 0.8 → odds = 0.8/0.2 = 4 ("4 to 1") → log-odds = ln 4 ≈ 1.386.

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

## Loss function: cross-entropy (log loss)
We do not use squared error here. For true label y in {0, 1} and predicted p:

    Loss = -[ y * ln(p) + (1 - y) * ln(1 - p) ]

Intuition:
- True y = 1, predicted p = 0.9 → loss = -ln(0.9) = **0.105** (small, good)
- True y = 1, predicted p = 0.1 → loss = -ln(0.1) = **2.303** (big, bad)
Confident wrong predictions are punished heavily.

Total loss over N points = average of the individual losses.
Minimizing it is the same as **maximum likelihood estimation**.

## Training with gradient descent
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

## Multi-class: Softmax
For K classes: P(class k | x) = e^(z_k) / Σ_j e^(z_j)

Example: scores z = (2, 1, 0) → exp = (7.39, 2.72, 1) → sum = 11.11 → probabilities = (0.665, 0.245, 0.090).

## Regularization
Add a penalty to prevent overfitting (especially with perfectly separable data, where weights grow forever):
- L2: Loss + λ * ||w||²
- L1: Loss + λ * ||w||₁ (can zero out features)

## Evaluation quick list
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC-AUC

## Key takeaways
- P(y=1|x) = σ(w · x + b)
- Log-odds are linear in x.
- Trained by minimizing cross-entropy.
- Output is a probability; the threshold (0.5 by default) is adjustable.