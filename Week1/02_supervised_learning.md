# Supervised Learning

## Definition

Learning a function `f: X → Y` from a dataset of (input, output) pairs, where the output is already known (labeled). The model's job is to generalize this mapping to new, unseen inputs.

## The two big tasks

- **Regression** — output is continuous (e.g., predicting a price, a temperature)
- **Classification** — output is discrete/categorical (e.g., spam vs not spam)

## How it works (high level)

1. Collect labeled training data `{(x1, y1), (x2, y2), ..., (xn, yn)}`
2. Choose a model/hypothesis class (linear model, decision tree, neural net, etc.)
3. Define a loss function that measures how wrong a prediction is
4. Optimize model parameters to minimize loss on training data
5. Check performance on unseen (test) data — this is what actually matters

## The Process (Training Set → Test Set flow)

- Data is split into a **Training Set** `{(X1,Y1), (X2,Y2), (X3,Y3), (X4,Y4), ...}` and a **Test Set** `{(X'1,Y'1), (X'2,Y'2), (X'3,Y'3), ...}`
- Example labeled points: `X1 = ⟨0.15, 0.25⟩, Y1 = -1` and `X2 = ⟨0.4, 0.45⟩, Y2 = +1`
- Flow:
  1. Training Set → **Training Algorithm** → produces a **Classifier**
  2. Test Set is fed into the Classifier for **Validation**
  3. Validation results loop back into the Training Algorithm to refine it
- **Why it matters**: this loop is exactly how you check generalization (from the "Key idea" below) — the Test Set is deliberately kept separate so validation reflects performance on unseen data, not memorized training data.

## Training (Agent view)

- Simplified block diagram: **Input (X)** → **Agent** → **Output (Ŷ)**
- The Agent's output Ŷ is compared to the actual **Target (Y)**
- The difference is the **Error** = Y − Ŷ (or similar), which is fed back into the Agent
- **Why it matters**: this is the mechanism behind step 3–4 of "How it works" — the loss function is essentially formalizing this Error signal, and "optimizing parameters" means adjusting the Agent until Error shrinks.

## Applications

- **Time series predictions** — e.g., rainfall in a certain region, spend on voice calls (regression over time)
- **Classification!** — the discrete-output task described above
- **Data reduction** — simplifying/compressing data while preserving useful structure
- **Trend analysis** — linear or exponential trends over data
- **Risk factor analysis** — identifying which factors contribute most to an output (interpretability angle)
- **Why it matters**: ties back to the two big tasks — time series and trend analysis lean regression-flavored, while risk factor analysis is about explaining *why* a model predicts what it predicts, not just *what* it predicts.

## Key idea I want to remember

Fitting training data well is not the goal — generalizing to new data is. A model that memorizes training data perfectly but fails on new data is useless (this connects directly to bias-variance later). The Training Set/Test Set split and Validation loop above is the concrete mechanism for checking this.

## My own example

Predicting whether a student will pass or fail based on attendance and assignment scores — that's classification. Predicting their exact marks out of 100 — that's regression.

## Questions to revisit

- How do you pick the right hypothesis class before knowing the true relationship in the data?
- How exactly does the Validation step's feedback get used to adjust the Training Algorithm (is this always gradient-based, or does it depend on the model)?