# Income Classification with Naive Bayes

Predicting whether an individual earns more or less than $50K/year from 1994 US Census data, comparing a fully supervised model against three semi-supervised extensions built on label propagation.

## What I did
Built a mixed Naive Bayes classifier (Gaussian for continuous features, Categorical for discrete features) to predict income class. Compared a standard supervised model against three semi-supervised approaches that incorporate an unlabelled dataset via pseudo-labelling, then evaluated which strategy generalised best.

## Data
- ~30K+ labelled training records, a separate unlabelled training set, and a held-out test set, all derived from the 1994 US Census
- 13 features spanning age, education, occupation, marital status, hours worked, capital gains/losses, and more
- Missing values dropped; categorical features ordinally encoded with unseen-category handling for the test set

## What I found
- The fully supervised model reached **81.2% test accuracy**
- Of three semi-supervised strategies tested (using all pseudo-labels, filtering to high-confidence pseudo-labels, and iterative two-stage label propagation), the confidence-filtered model performed best but still only reached **79.8%** — it could not beat the supervised baseline
- Education, occupation, and relationship status were the strongest categorical predictors of income; education-num, age, and hours-per-week were the strongest continuous predictors
- Treating pseudo-labels as ground truth introduced measurable confirmation bias, increasing false positives/negatives relative to the supervised model

## What it means
Semi-supervised learning is often assumed to improve performance by making better use of available data, but this project shows that isn't guaranteed: when the labelled dataset is already reasonably sized and clean, adding pseudo-labelled data can reinforce the model's own mistakes instead of reducing uncertainty. This highlights the importance of testing whether semi-supervised techniques are actually worth the added complexity, rather than assuming they will outperform a well-tuned supervised model.

## Tech stack
Python, scikit-learn (GaussianNB, CategoricalNB, OrdinalEncoder), pandas, NumPy, matplotlib

## Repo structure
```
notebooks/       # Full analysis notebook
report/          # Written report (PDF) with extended discussion and figures
requirements.txt
```
