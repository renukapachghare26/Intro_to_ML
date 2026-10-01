# Tutorial Notes – Linear Classification, Logistic Regression and LDA

## Quick comparison
| Method | Models | Output | Boundary | Needs Gaussian assumption? |
|---|---|---|---|---|
| Perceptron | Direct boundary | Class only | Linear | No |
| Logistic regression | P(y|x) | Probability | Linear | No |
| LDA | P(x|y) and priors | Probability | Linear | Yes (shared Σ) |
| QDA | P(x|y), each class Σ | Probability | Quadratic | Yes |

## Formula cheat sheet
- Linear score: z = wᵀx + b
- Sigmoid: σ(z) = 1 / (1 + e^(-z))
- Log-odds: ln(p/(1-p)) = wᵀx + b
- Cross-entropy: -[y ln p + (1-y) ln(1-p)]
- Gradient: (1/N) Σ (p_i - y_i) x_i
- Fisher criterion: J(w) = (wᵀ S_B w)/(wᵀ S_W w)
- LDA direction (2 classes): w ∝ S_W⁻¹ (μ2 - μ1)
- LDA discriminant: δ_k(x) = xᵀΣ⁻¹μ_k - ½μ_kᵀΣ⁻¹μ_k + ln π_k
- Max LDA axes: min(K - 1, d)

## Practice questions with answers

**Q1.** w = (3, -1), b = -2. Classify x = (2, 1).
score = 3*2 + (-1)*1 - 2 = 3 → positive → Class 1.

**Q2.** For logistic regression, z = 0. What is P(y=1)?
σ(0) = 0.5.

**Q3.** If w = 0.7 for "hours studied", how do odds change per extra hour?
Odds multiply by e^0.7 ≈ 2.01, so they roughly double.

**Q4.** Why is squared error a bad loss for logistic regression?
It makes the loss non-convex with the sigmoid, and gives weak gradients for confident wrong predictions.
Cross-entropy is convex and punishes confident mistakes strongly.

**Q5.** In LDA with 4 classes and 10 features, how many discriminant directions?
min(4 - 1, 10) = 3.

**Q6.** Class means 3 and 7 (1D), equal variance, equal priors. Where is the LDA boundary?
(3 + 7)/2 = 5.

**Q7.** What happens to the boundary if class 2 (mean 7) becomes much rarer?
It shifts toward 7 (the rarer class).

**Q8.** Difference between PCA and LDA?
PCA is unsupervised and maximizes variance. LDA is supervised and maximizes class separation.

## Common mistakes
- Forgetting the bias term b.
- Thinking logistic regression is a regression method. It is a classifier.
- Using LDA when classes have very different covariances (use QDA).
- Not scaling features before gradient descent.
- Forgetting that LDA gives at most K - 1 axes.
- Treating 0.5 as the only possible threshold. It can be tuned (for example lower it when missing a positive is costly).

## Suggested study order
1. 01 (geometry of a linear boundary)
2. 02 (probabilities, loss, training)
3. 03 → 04 (Fisher's idea and the math)
4. 05 (the Bayes view and how it connects back to 02)
5. Run `hours_vs_marks.py` and compare the three boundaries.