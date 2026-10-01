      # 01 – Linear Classification

## What is it?
Classification means predicting a **category** (pass/fail, spam/not spam) instead of a number.
A **linear classifier** separates classes using a straight line (2D), a plane (3D), or a hyperplane (higher dimensions).

## The model
For input features x = (x1, x2, ..., xd):

    score(x) = w · x + b = w1*x1 + w2*x2 + ... + wd*xd + b

- **w** = weight vector (decides the direction of the boundary)
- **b** = bias (shifts the boundary away from the origin)

Prediction rule:
- score(x) > 0  → Class 1
- score(x) < 0  → Class 0 (or -1)
- score(x) = 0  → the **decision boundary**

## Geometry (important)
- The boundary w · x + b = 0 is a line/plane.
- **w is perpendicular (normal) to the boundary** and points toward the Class 1 side.
- Distance of a point x from the boundary = (w · x + b) / ||w||
- Bigger |score| means the point is farther from the boundary, so we are more confident.

## Simple example
Predict **Pass/Fail** from two features: hours studied (x1) and hours slept (x2).

Suppose we learned: w = (2, 1), b = -10

| Student | x1 (study) | x2 (sleep) | score = 2*x1 + x2 - 10 | Prediction |
|---|---|---|---|---|
| A | 5 | 4 | 2*5 + 4 - 10 = **4** | Pass |
| B | 2 | 3 | 2*2 + 3 - 10 = **-3** | Fail |
| C | 4 | 2 | 2*4 + 2 - 10 = **0** | On the boundary |

The boundary is the line 2*x1 + x2 = 10.

## 1D example
Only hours studied (x), with w = 1, b = -4:
- score = x - 4
- x = 6 → score = 2 → Pass
- x = 2 → score = -2 → Fail
- The boundary is at x = 4 hours.

## How do we learn w and b?
Several methods exist, which the next notes cover:
1. **Perceptron**: fix mistakes one by one.
2. **Logistic regression** (note 02): a probabilistic approach.
3. **LDA** (notes 03–05): use class means and covariance.
4. **SVM**: maximize the margin (not covered here).

### Perceptron update (quick idea)
If a point (x, y) with y in {+1, -1} is misclassified:

    w = w + y * x
    b = b + y

Example: w = (0,0), b = 0, point x = (2,1) with y = +1.
score = 0 → counted as a mistake → w = (2,1), b = 1.
Now score(x) = 2*2 + 1*1 + 1 = 6 > 0, correct.

## Linearly separable vs not
- **Separable**: a line can perfectly split the classes. The perceptron converges.
- **Not separable**: classes overlap or form a pattern like XOR. A straight line cannot perfectly separate them, so we need soft methods (logistic regression) or non-linear features.

## Why not just use linear regression for classes?
You can (encode classes as 0/1 and fit a line), but:
- Outputs can be below 0 or above 1, so they are not probabilities.
- Extreme points (outliers) can drag the line and hurt the boundary.
This is why logistic regression exists (next file).

## Key takeaways
- Linear classifier = sign of (w · x + b).
- w is perpendicular to the boundary.
- Score magnitude is proportional to the distance from the boundary.
- Works well when classes are roughly separable by a straight boundary.