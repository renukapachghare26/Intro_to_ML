# 03 – Linear Discriminant Analysis (LDA) I: Introduction

## Recap: where we are
- Three discriminant-based approaches for linear classification:
  1. Linear regression on an indicator variable
  2. Logistic regression
  3. **LDA** (this note)
- Logistic regression assumption: the **log-odds** ln( P(class 1) / P(class 0) ) is a **linear function** of x.
  - Individual probabilities are sigmoids, but the log-odds is linear.
  - That gives a **linear decision boundary**.

## Name warning
- In ML, **LDA** can mean two different things:
  - **Linear Discriminant Analysis**: the classifier in these notes.
  - **Latent Dirichlet Allocation**: a different method for modelling distributions (topic models). It has nothing to do with classification.
- Be context-sensitive when you see "LDA".

## What is LDA?
LDA is a method that finds a direction (line) onto which, when we **project** the data,
the classes are **as well separated as possible**. It is used for both:
1. **Classification**, and
2. **Dimensionality reduction** (supervised, since it uses class labels).

Compare with PCA: PCA finds the direction of max variance (ignores labels).
LDA finds the direction of max **class separation** (uses labels).
(Think of it as like principal component regression, but taking the class labels into account.)

## The Bayes' rule view (probabilistic setup)
- What we really want: **P(class k | x)**, the probability of the class given the data point.
- Bayes' rule:

      P(G = k | x) = f_k(x) · π_k / P(x)

- Ingredients:
  - **f_k(x) = P(x | class = k)**: the **class-conditional density** of x.
  - **π_k = P(class = k)**: the **prior** probability of class k. All priors sum to 1 (every point belongs to some class).
  - **P(x)**: we do not need to model it separately. Marginalize over classes:

        P(x) = Σ_l f_l(x) · π_l

- So:

      P(G = k | x) = f_k(x) · π_k / Σ_{l=1}^{K} f_l(x) · π_l

- The only real modelling choice is **the form of f_k(x)**. Different assumptions on f_k give different classifiers.

## Common assumptions on f_k
1. **Single multivariate Gaussian per class**
   - **LDA**: Gaussian classes sharing the same covariance matrix.
   - **QDA** (Quadratic Discriminant Analysis): Gaussian classes, each with its own covariance.
2. **Mixture of Gaussians per class**
3. **Non-parametric densities**
4. **Naive Bayes assumption** (covered in a later class)

### Mixture distributions
- Idea: if a distribution looks too complex for one simple form, write it as a **weighted combination of several simpler distributions**.
- 1D example: a density with two peaks can be built from two Gaussians, suitably weighted.
- 2D example: a class has two dense regions with a low-density gap between them.
  - A **single Gaussian** (fitted by maximum likelihood) puts its peak in the gap, which is wrong.
  - **Two Gaussians per class**, combined with weights, put peaks at the real dense regions.
- Can use more components (e.g. 10) or even non-Gaussian forms, but more complex forms make the problem **harder to solve**.
- Choosing the number of components is a hard problem. Options:
  - Use domain knowledge.
  - Run preliminary experiments, e.g. rough clustering while varying the number of clusters.

### Parametric vs non-parametric
- **Parametric**: the number of parameters is fixed a priori (e.g. "2 Gaussians per class").
- **Non-parametric**: the number of parameters is **unbounded**, not zero.
  - The model can add parameters (e.g. more Gaussians) **if the data warrants it**.
  - Need care about **overfitting**.
- (Not covered in this intro course; belongs to an advanced ML topic.)

### Naive Bayes assumption (preview)
- Factor the class-conditional density along each dimension:

      P(x | k) = P(x1 | k) · P(x2 | k) · ...

- Meaning: **given the class**, features are independent of each other.
- Very strong (naive) assumption. Without knowing the class, x1 and x2 may look dependent, but given k we assume they are not.
- Still works well in many settings. Covered separately later.

### Multivariate Gaussian (the assumption used for LDA and QDA)

    f_k(x) = 1 / ( (2π)^(p/2) |Σ_k|^(1/2) ) · exp( −½ (x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k) )

- μ_k = mean vector of class k, Σ_k = covariance matrix of class k.
- Univariate case: the variance σ² takes the place of Σ; (x − μ)²/σ² becomes (x − μ)ᵀ Σ⁻¹ (x − μ) in the vector case.
- In 2D this is a bell-shaped surface over the (x1, x2) plane.

## Core idea (Fisher's criterion)
Project each point onto a line with direction w:  y = wᵀx

We want a w such that, after projection:
- The **class means are far apart** (large between-class distance), and
- Each class is **tightly clustered** (small within-class spread).

## Key quantities
For two classes with means μ1, μ2:

- **Within-class scatter**: S_W = S_1 + S_2, where S_k = Σ (x - μ_k)(x - μ_k)ᵀ over points in class k
- **Between-class scatter**: S_B = (μ2 - μ1)(μ2 - μ1)ᵀ

## Worked example (2D, two classes)
Class 1: (4,2), (2,4), (2,3), (3,6), (4,4)
Class 2: (9,10), (6,8), (9,5), (8,7), (10,8)

**Step 1: means**
- μ1 = (3, 3.8)
- μ2 = (8.4, 7.6)

**Step 2: within-class scatter**
- S_1 = [[4, -1], [-1, 8.8]]
- S_2 = [[9.2, -0.2], [-0.2, 13.2]]
- S_W = S_1 + S_2 = [[13.2, -1.2], [-1.2, 22]]

**Step 3: direction** (derived in note 04): w ∝ S_W⁻¹ (μ2 - μ1)
- μ2 - μ1 = (5.4, 3.8)
- S_W⁻¹ = (1/288.96) * [[22, 1.2], [1.2, 13.2]]
- w ≈ (0.427, 0.196)

**Step 4: project the means**
- Class 1 mean → 0.427*3 + 0.196*3.8 ≈ **2.03**
- Class 2 mean → 0.427*8.4 + 0.196*7.6 ≈ **5.08**
- Threshold = midpoint ≈ **3.55**

**Step 5: classify a new point** (6, 8):
projection = 0.427*6 + 0.196*8 ≈ 4.13 > 3.55 → **Class 2** ✓

## Assumptions of LDA
- Each class is roughly Gaussian (bell-shaped).
- All classes share the **same covariance matrix**.
- Features are not wildly collinear.

## When LDA works well / struggles
Works well:
- Classes are roughly normal with similar spread
- Small datasets (fewer parameters than QDA)
Struggles:
- Different covariance per class → consider QDA
- Non-linear boundaries
- Heavy outliers
- Class densities that are multi-peaked (a single Gaussian per class fits badly) → consider mixtures

## Key takeaways
- LDA is the third discriminant-based linear classifier (after indicator regression and logistic regression).
- Probabilistic view: P(k | x) = f_k(x) π_k / Σ_l f_l(x) π_l, with f_k the class-conditional density and π_k the prior.
- Different assumptions on f_k give different classifiers: Gaussian (LDA, QDA), mixtures, non-parametric, naive Bayes.
- LDA and QDA assume a single multivariate Gaussian per class; LDA shares one covariance across classes.
- Geometric view: find the projection that maximizes class separation relative to the spread inside classes.
- S_W (within) should be small, S_B (between) should be large.
- Result is a linear decision boundary, like the models in notes 01 and 02.