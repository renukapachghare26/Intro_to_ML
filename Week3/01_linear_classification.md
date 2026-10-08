# 01 – Linear Classification

## What is it?
- Classification = predicting a **category** (pass/fail, spam/not spam) instead of a number.
- **Linear regression**: the response is a linear function of the inputs.
- **Linear classification**: the **boundary** between classes is linear (a line in 2D, a plane in 3D, a hyperplane in higher dimensions).
- Non-linear *regression* is still possible with basis transformations; here we only restrict the **separating surface** to be a hyperplane.

## Setup (notation)
- Input: x ∈ R^p (add a leading 1 to x for the bias, so it becomes p+1 dimensional).
- Output: class label G from a set of K classes, indexed 1 to K.
- Training labels use **1-of-K encoding**: K indicator variables Y1 ... YK, and exactly one is 1 for each data point.

## Two families of approaches
1. **Discriminant functions**: learn one function δ_k(x) per class and pick the highest.
   - Methods: linear regression on indicators, logistic regression, LDA.
2. **Directly model the hyperplane**: skip class-wise functions.
   - Methods: perceptron, and the "optimal hyperplane" idea (leads to SVM).
- If you have the discriminant functions you can always recover the hyperplane.

## Discriminant functions
- Rule: classify x as class i if δ_i(x) is greater than all other δ_k(x).
- Two-class case: boundary is where δ1(x) = δ2(x).
  - δ1 > δ2 → class 1
  - δ2 > δ1 → class 2
- When is the boundary linear?
  - **Sufficient**: δ1 and δ2 are linear.
  - **Not necessary**: δ's can be non-linear if some monotone transformation of them is linear.
  - So a model may look heavily non-linear and still give a linear boundary (seen later, e.g. in LDA / logistic regression).

## The model (geometric view)
For x = (x1, ..., xd):

    score(x) = w · x + b = w1*x1 + ... + wd*xd + b

- **w** = weight vector (direction of the boundary)
- **b** = bias (shifts the boundary from the origin)
- score(x) > 0 → Class 1
- score(x) < 0 → Class 0 (or -1)
- score(x) = 0 → the **decision boundary**

### Geometry (important)
- w is **perpendicular (normal)** to the boundary and points toward the Class 1 side.
- Distance of x from the boundary = (w · x + b) / ||w||
- Bigger |score| → farther from the boundary → more confident.

## Linear regression on an indicator matrix
- Build the response matrix Y (N × K), one indicator column per class.
- Fit all columns at once:

      B_hat = (X^T X)^-1 X^T Y

  - B is a **matrix** (capital beta): one column of coefficients per class.
- For a new x (with 1 prepended): f(x) = [1, x]·B_hat → a vector of K outputs.
- Predicted class = **argmax_k f_k(x)**.
- No complex math: it is the same least-squares fit as before.

### What does f(x) mean?
- Regression predicts the **expected value** E[Y_k | x].
- For an indicator, E[Y_k | x] = P(class = k | x).
  - Imagine seeing the same x many times: Y_k = 1 when it is class k, 0 otherwise. The average is the probability.
- So ideally f_k(x) ≈ P(G = k | x).
- **Problem**: linear regression is unconstrained, so f_k(x) can be < 0 or > 1.
  - Cannot be read directly as a probability.
  - Fix comes later (logistic regression).
- It is still useful: whichever output is **largest** tells you the class, even if the values are not probabilities.
- People still use it because it is easy to do.

## Masking problem (big pitfall)
- Example: 1D input, three classes (pink, blue, brown) with brown **in the middle**.
- Each indicator is fit by a straight line:
  - Pink and blue lines slope across the input range.
  - Brown's line is nearly flat/low (lots of 0s on both sides, few 1s in the middle).
- Result: brown's output is **never the highest** anywhere → class brown is **never predicted**.
- This is called **masking**: a middle class is hidden by the others.

### Fixing masking
- Use **higher-order basis transformations**: regress on x, x² (and so on) instead of only x.
- With x² (quadratic):
  - Curves cross at two points; blue on one side, brown in between, pink on the other side.
  - Boundaries are recovered correctly.
- **Rule of thumb**: with K classes you need polynomial terms up to degree **K − 1**.
  - 3 classes → quadratic is enough.
  - 4 classes → quadratic still masks, need cubic.
- Note: the input is just a line segment; the "regions" are intervals on that line.

## Simple example
Predict **Pass/Fail** from hours studied (x1) and hours slept (x2).
Learned: w = (2, 1), b = -10

| Student | x1 (study) | x2 (sleep) | score = 2*x1 + x2 - 10 | Prediction |
|---|---|---|---|---|
| A | 5 | 4 | 2*5 + 4 - 10 = **4** | Pass |
| B | 2 | 3 | 2*2 + 3 - 10 = **-3** | Fail |
| C | 4 | 2 | 2*4 + 2 - 10 = **0** | On the boundary |

Boundary: 2*x1 + x2 = 10.

## 1D example
Only hours studied (x), w = 1, b = -4:
- score = x - 4
- x = 6 → score = 2 → Pass
- x = 2 → score = -2 → Fail
- Boundary at x = 4 hours.

## How do we learn w and b?
1. **Linear regression on indicators** (above).
2. **Logistic regression** (note 02): probabilistic approach.
3. **LDA** (notes 03–05): uses class means and covariance. Like principal component regression, but it **uses the class labels** to find directions for classification.
4. **Perceptron**: fix mistakes one by one.
5. **SVM / optimal hyperplane**: maximize the margin (not covered here).

### Perceptron update (quick idea)
If (x, y) with y in {+1, -1} is misclassified:

    w = w + y * x
    b = b + y

Example: w = (0,0), b = 0, point x = (2,1), y = +1.
- score = 0 → counted as a mistake → w = (2,1), b = 1.
- New score = 2*2 + 1*1 + 1 = 6 > 0 → correct.

## Linearly separable vs not
- **Separable**: a line can perfectly split the classes. The perceptron converges.
- **Not separable**: classes overlap or form XOR-like patterns. Need soft methods (logistic regression) or non-linear features.

## Why not just use linear regression for classes?
- Outputs can fall below 0 or above 1 → not probabilities.
- Outliers can drag the line and hurt the boundary.
- **Masking** with 3 or more classes.
- This is why logistic regression exists (next file).

## Key takeaways
- Linear classifier = sign of (w · x + b); boundary is a hyperplane.
- Two approaches: discriminant functions vs directly modelling the hyperplane.
- Boundary is linear if the δ's are linear, or a monotone transform of them is.
- Indicator regression: B_hat = (X^T X)^-1 X^T Y, predict by argmax.
- f_k(x) estimates P(G=k | x) but is not constrained to [0, 1].
- Masking hides middle classes; K classes need degree K − 1 terms.
- w is perpendicular to the boundary; |score| ∝ distance from it.
- Works well when classes are roughly separable by a straight boundary.