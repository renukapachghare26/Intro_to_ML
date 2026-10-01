# 03 – Linear Discriminant Analysis (LDA) I: Introduction

## What is LDA?
LDA is a method that finds a direction (line) onto which, when we **project** the data,
the classes are **as well separated as possible**. It is used for both:
1. **Classification**, and
2. **Dimensionality reduction** (supervised, since it uses class labels).

Compare with PCA: PCA finds the direction of max variance (ignores labels).
LDA finds the direction of max **class separation** (uses labels).

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

## Key takeaways
- LDA = find the projection that maximizes class separation relative to the spread inside classes.
- S_W (within) should be small, S_B (between) should be large.
- Result is a linear decision boundary, like the models in notes 01 and 02.