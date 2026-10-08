# SVM Kernels (The Kernel Trick)

## Starting observation
- In the SVM **dual**, the data points appear only through **inner products** x_iᵀx_j.
- In the final **classifier** f(x) = Σ α_i y_i x_iᵀx + b, the data again appears only through inner products.
- So to train and to use an SVM, we only need to be able to compute inner products between pairs of points.
- If we find an **efficient way of computing these inner products**, we can do something interesting.

## Basis expansion + SVM
- To make linear classifiers more powerful we normally use **basis transformations**: replace x with h(x).
  - Example: replace x with x² (or add x² as an extra feature). This gives a larger basis.
- The SVM math goes through unchanged, with x replaced by h(x). The dual becomes:

      maximize   Σ_i α_i − ½ Σ_i Σ_j α_i α_j y_i y_j ⟨h(x_i), h(x_j)⟩
      subject to 0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0

- The classifier becomes:

      f(x) = Σ_{i ∈ support vectors} α_i y_i ⟨h(x), h(x_i)⟩ + b

- So we solve **the same kind of optimization problem**, but in a **transformed space**.
- What we need to know is only ⟨h(x), h(x′)⟩ for pairs of points:
  - **Training**: pairs of training points.
  - **Prediction**: a support vector and the new input.

## Kernels
- Define the **kernel function**:

      K(x, x′) = ⟨h(x), h(x′)⟩

- A kernel is a **similarity measure** (a kind of distance function) between two points. **Kernels are nothing but similarity functions.**
- The key property: K **operates on x and x′** (the original points) but **returns the inner product of h(x) and h(x′)**.
- So you never have to compute h(x) explicitly. This is the **kernel trick**.
- Dual and classifier with kernels:

      maximize   Σ_i α_i − ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
      f(x) = Σ_{i ∈ SV} α_i y_i K(x, x_i) + b

## What makes a valid kernel?
- K must be **symmetric**: K(x, x′) = K(x′, x).
- K must be **positive (semi-)definite**.
  - Recall: a matrix A is positive definite if the quadratic form xᵀAx is positive for non-zero x. (Semi-definite: ≥ 0.)
  - Mechanical reason: the SVM optimization involves a **quadratic form**. If it could become negative, the optimization would break down badly.
  - There is a much deeper reason (covered in a kernel methods course, not here).

## Popular kernels
| Kernel | Formula | Parameters to tune |
|---|---|---|
| **Polynomial** | K(x, x′) = (1 + ⟨x, x′⟩)^d | d (degree), C |
| **Gaussian / RBF** | K(x, x′) = exp(−γ ‖x − x′‖²) | γ (width), C |
| **Sigmoid** (neural network kernel) | K(x, x′) = tanh(κ1 ⟨x, x′⟩ + κ2) | κ1, κ2, C |

- **Polynomial**: d = 1 gives the plain linear SVM we derived so far. d = 2, 3, 4, ... give more flexible boundaries.
- **RBF**: a Gaussian without the normalizing factor, hence "RBF kernel". (To call it the Gaussian kernel strictly, you would need the normalization.)
- **Sigmoid**: uses the hyperbolic tangent; κ1 and κ2 are arbitrary constants you tune.
- These work on generic data. For specific data types, people design **specialized kernels**, e.g. many **string kernels** for measuring similarity between strings.

### Kernels work beyond real vectors
- So far x was assumed to come from R^p.
- With a proper kernel, max-margin classification applies to **any kind of data** (strings, etc.), as long as you can define a valid similarity function.
- This is **not** true of many other methods you have seen, which depend on the data being real-valued.

## Worked example: polynomial kernel, degree 2, two features
- Take p = 2 (x = (x1, x2)), d = 2.
- Expand the kernel:

      K(x, x′) = (1 + x1x1′ + x2x2′)²
               = 1 + 2x1x1′ + 2x2x2′ + x1²x1′² + x2²x2′² + 2x1x1′x2x2′

- This equals ⟨h(x), h(x′)⟩ for the **quadratic basis expansion**:

      h(x) = (1, √2·x1, √2·x2, x1², x2², √2·x1x2)

  - (First coordinate 1; then the original features; then the squares and the cross term. The √2 factors make the inner product match the expansion exactly.)
- So a 2D point is mapped into a **6-dimensional** space.
- **Numerical check**: x = (2, 3), x′ = (4, 5).
  - Direct kernel: ⟨x, x′⟩ = 8 + 15 = 23 → (1 + 23)² = 24² = **576**.
  - Through h: h(x) = (1, 2√2, 3√2, 4, 9, 6√2), h(x′) = (1, 4√2, 5√2, 16, 25, 20√2).
    - Inner product = 1 + 16 + 30 + 64 + 225 + 240 = **576** ✓.
- Same answer, but the kernel computes it using only the **original 2D vectors**.
- (The lecture's spoken example gave "22 squared"; with these numbers the inner product is 23, so the correct value is 24² = 576.)

### Why this matters for bigger degrees
- Number of polynomial features of degree up to d in p dimensions = C(p + d, d).
  - p = 2, d = 2 → 6 ✓.
  - p = 2, d = 15 → 136 features.
- With the kernel you do **about the same work regardless of d**: compute the inner product (cost ∝ p), add 1, and raise to the power d.
- You skip building the huge expanded vectors.

## RBF kernel: computing in an infinite-dimensional space
- Worked numbers (γ = 0.5):
  - x = x′ → K = e⁰ = **1** (identical points, maximum similarity).
  - x = (0,0), x′ = (1,1): ‖x − x′‖² = 2 → K = e^(−1) ≈ **0.37**.
  - Larger γ = 2 on the same pair: K = e^(−4) ≈ **0.018** (almost "not similar").
- As the distance grows, K → 0. So K behaves like a similarity score between 0 and 1.
- **γ controls the width** of the Gaussian: large γ → narrow, only very close points are similar; small γ → wide.
- Interpretation: the RBF kernel corresponds to a basis expansion into an **infinite-dimensional** vector space. It is not even easy to write h(x) down.
- Yet all computation happens in the **original space**.
  - Polynomial example: computed in R² but the result equals a computation in R⁶.
  - RBF: computed in R^p but the result corresponds to an infinite-dimensional space.
- This is why it is called the **kernel trick**, and why RBF kernels are considered powerful and work on a wide variety of data.
- But they are **not all-powerful**. Be careful when using them.

## Linear in feature space, non-linear in input space
- The SVM still finds a **separating hyperplane**, but in the h(x) space.
- Mapped back to the original x space, that boundary is generally **non-linear** (and can be very complex with an RBF kernel).

## Parameters you tune in practice
- **RBF**: **C** (penalty on the slack / margin violations) and **γ** (width of the Gaussian).
- **Polynomial**: **d** and **C**.
- **Sigmoid**: **κ1, κ2** and **C**.
- This form of SVM is called **C-SVM**.

## ν-SVM (other ways to constrain)
- Besides penalizing the slack variables (C), you can control something else: the **number of support vectors**.
- Problem: with RBF kernels the separating hyperplane can be very complex, so you can end up with **a very large fraction of the data as support vectors**.
  - With a very high C, it is common to see something like **60%** of the data as support vectors.
  - That is not interesting. Linear boundaries would not do this (not all points can be equally far from the hyperplane), but RBF boundaries can.
- Instead of tuning C and γ again and again to reduce the count, use **ν-SVM** (the Greek letter nu, not "new").
  - Lets you specify a bound on the number of support vectors ("do the best you can, but not more than about 30").
  - (Strictly, ν bounds the fraction of margin errors from above and the fraction of support vectors from below. Check your package's documentation for the exact meaning.)

## Summary of the whole SVM story
1. **Hard-margin** SVM: maximize the margin for linearly separable data (notes 07 and 08).
2. **Soft-margin** SVM: slack variables ξ_i and penalty C for inseparable data.
3. **Kernels**: replace inner products with K(x, x′) to get non-linear boundaries without explicit basis expansion.
4. Parameters to tune: C (+ γ, d, κ's depending on the kernel); optionally ν.

## Key takeaways
- The dual and the classifier only need **inner products**, so we can swap ⟨x, x′⟩ for a kernel K(x, x′) = ⟨h(x), h(x′)⟩.
- Kernels are **similarity functions** that compute inner products in the transformed space **without ever forming h(x)** (the kernel trick).
- A valid kernel is **symmetric and positive semi-definite**.
- Polynomial: (1 + ⟨x, x′⟩)^d. RBF: exp(−γ‖x − x′‖²). Sigmoid: tanh(κ1⟨x, x′⟩ + κ2).
- Degree-2 polynomial in 2D = quadratic expansion in 6D; the kernel gave 576 for x = (2,3), x′ = (4,5) either way.
- The RBF kernel works in an **infinite-dimensional** space, which makes it very powerful.
- Kernels let SVMs work on **non-vector data** (e.g. strings) given a proper similarity function.
- Tune **C and γ** (RBF), **C and d** (polynomial), **C, κ1, κ2** (sigmoid). Use **ν-SVM** to control the number of support vectors.