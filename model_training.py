#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)


# ============================================================
# 1. Loading Data & Building features
# ============================================================
print("=" * 70)
print("1. Load Data & Build Features")
print("=" * 70)

df = pd.read_csv("cleaned_data.csv")
y = df['target'].values
CLASS_NAMES = ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']

# --- 1.1 Structured Features (22 dimensions) ---
num_cols = ['emotion_intensity', 'sombre', 'content', 'calm', 'uneasy',
            'num_colours', 'num_objects', 'willing_to_pay']
binary_cols = [c for c in df.columns if c.startswith(('season_', 'room_', 'view_'))]
struct_cols = num_cols + binary_cols

X_struct_raw = df[struct_cols].values

# MinMaxScaler normalization (LogReg sensitive to feature scale)
scaler = MinMaxScaler()
X_struct = scaler.fit_transform(X_struct_raw)

print(f"  Structured Features: {X_struct.shape[1]} dimensions")
print(f"    Numerical: {len(num_cols)} columns — {num_cols}")
print(f"    Binary: {len(binary_cols)} columns")

# --- 1.2 Text Features (TF-IDF) ---
text_configs = {
    'feeling_desc': 150,   # Emotional description, most vocabulary, most dimensions
    'food':          50,   # Food associations, limited vocabulary
    'soundtrack':    80,   # Music description
}
tfidf_models = {}
X_text_parts = []

for col, max_feat in text_configs.items():
    text_data = df[col].fillna('')
    tfidf = TfidfVectorizer(
        max_features=max_feat,
        stop_words='english',
        min_df=5               # At least appear 5 times to keep, filter noise words
    )
    X_part = tfidf.fit_transform(text_data).toarray()
    tfidf_models[col] = tfidf
    X_text_parts.append(X_part)
    print(f"  TF-IDF [{col}]: {X_part.shape[1]} dimensions (max_features={max_feat}, min_df=5)")

X_text = np.hstack(X_text_parts)
print(f"  Text Features Combined: {X_text.shape[1]} dimensions")

# --- 1.3 Merge Features ---
X_combined = np.hstack([X_struct, X_text])
print(f"  Combined Feature Matrix: {X_combined.shape} (struct {X_struct.shape[1]} + text {X_text.shape[1]})")

# --- 1.4 Uniform CV Split ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"  Evaluation Method: 5-fold Stratified CV (random_state=42)")


# ============================================================
# 2. Utility Functions
# ============================================================
def run_experiment(name, model, X):
    """
    Run cross_validate on given features, return:
      (model_name, train_acc, cv_acc, cv_std, gap)
    """
    res = cross_validate(model, X, y, cv=cv, scoring='accuracy',
                         return_train_score=True)
    train_acc = res['train_score'].mean()
    cv_acc    = res['test_score'].mean()
    cv_std    = res['test_score'].std()
    gap       = train_acc - cv_acc
    return (name, train_acc, cv_acc, cv_std, gap)


def print_table(results, title):
    """Format and print experiment results table"""
    print(f"\n{title}")
    header = f"  {'Model':<45s} | {'Train':>7s} | {'CV Acc':>14s} | {'Gap':>7s}"
    print(header)
    print("  " + "-" * len(header))
    for name, tr, cv_acc, cv_std, gap in results:
        print(f"  {name:<45s} | {tr:>6.2%} | {cv_acc:.2%} ± {cv_std:.2%} | {gap:>6.2%}")


# ============================================================
# 3. Model Family 1: Logistic Regression (Linear Models)
# ============================================================
print("\n" + "=" * 70)
print("2. Model Training — Family 1: Logistic Regression")
print("=" * 70)
print("  L2 regularization, hyperparameter C controls regularization strength (C larger means weaker regularization)")

results_lr = []

# Structured features only
for C in [0.01, 0.1, 1.0, 10.0]:
    r = run_experiment(f'LogReg C={C:<5} [struct 22d]',
                       LogisticRegression(C=C, max_iter=2000), X_struct)
    results_lr.append(r)

# Combined features
for C in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
    r = run_experiment(f'LogReg C={C:<5} [combined 302d]',
                       LogisticRegression(C=C, max_iter=3000), X_combined)
    results_lr.append(r)

print_table(results_lr, "--- Logistic Regression Experiment Results ---")

# Find optimal C on combined features
best_lr = max([r for r in results_lr if 'combined' in r[0]], key=lambda x: x[2])
print(f"\n  ★ LogReg optimal: {best_lr[0]}, CV={best_lr[2]:.2%}")


# ============================================================
# 4. Model Family 2: Naive Bayes (Probabilistic Models)
# ============================================================
print("\n" + "=" * 70)
print("3. Model Training — Family 2: Naive Bayes")
print("=" * 70)
print("  Gaussian NB for continuous features, Multinomial NB for TF-IDF text features")

results_nb = []

# Gaussian NB — Structured features
results_nb.append(run_experiment(
    'Gaussian NB [struct 22d]', GaussianNB(), X_struct))

# Multinomial NB — Text features (requires non-negative input, TF-IDF satisfies)
results_nb.append(run_experiment(
    'Multinomial NB [text 280d]', MultinomialNB(), X_text))

# Multinomial NB + alpha tuning
for alpha in [0.1, 0.5, 1.0, 2.0]:
    results_nb.append(run_experiment(
        f'Multinomial NB alpha={alpha} [text 280d]',
        MultinomialNB(alpha=alpha), X_text))

# Multinomial NB — Combined features (struct normalized to [0,1], non-negative, usable)
results_nb.append(run_experiment(
    'Multinomial NB [combined 302d]', MultinomialNB(), X_combined))

print_table(results_nb, "--- Naive Bayes Experiment Results ---")

best_nb = max(results_nb, key=lambda x: x[2])
print(f"\n  ★ NB optimal: {best_nb[0]}, CV={best_nb[2]:.2%}")


# ============================================================
# 5. Model Family 3: Decision Tree (Tree Models)
# ============================================================
print("\n" + "=" * 70)
print("4. Model Training — Family 3: Decision Tree")
print("=" * 70)
print("Hyperparameter max_depth controls tree complexity")

results_dt = []

# Structured features — depth scan
for depth in [3, 5, 7, 10, 15, None]:
    d_str = str(depth) if depth else 'None'
    results_dt.append(run_experiment(
        f'DTree depth={d_str:<5} [struct 22d]',
        DecisionTreeClassifier(max_depth=depth, random_state=42), X_struct))

# Combined features
for depth in [5, 7, 10, 15]:
    results_dt.append(run_experiment(
        f'DTree depth={depth:<5} [combined 302d]',
        DecisionTreeClassifier(max_depth=depth, random_state=42), X_combined))

print_table(results_dt, "--- Decision Tree Experiment Results ---")

best_dt = max(results_dt, key=lambda x: x[2])
print(f"\n  ★ DTree best: {best_dt[0]}, CV={best_dt[2]:.2%}")


# ============================================================
# 6. Three-race horizontal comparison & final selection
# ============================================================
print("\n" + "=" * 70)
print("5. Three-race horizontal comparison")
print("=" * 70)

finalists = [best_lr, best_nb, best_dt]
print_table(finalists, "--- Best models from each family ---")

overall_best = max(finalists, key=lambda x: x[2])
print(f"\n  ★★★ Final selection: {overall_best[0]}")
print(f"       CV Accuracy = {overall_best[2]:.2%} ± {overall_best[3]:.2%}")
print(f"       Train-CV Gap = {overall_best[4]:.2%}")

# Analysis of selection rationale
print(f"\n  Selection rationale:")
print(f"    1. CV accuracy highest ({overall_best[2]:.2%})")
print(f"    2. Train-CV gap = {overall_best[4]:.2%}, "
      f"{'controllable overfitting' if overall_best[4] < 0.06 else 'some overfitting, but accuracy advantage is significant'}")
print(f"    3. Logistic Regression inference requires only matrix multiplication, satisfying pred.py numpy restriction")


# ============================================================
# 7. Hold-out test + detailed evaluation
# ============================================================
print("\n" + "=" * 70)
print("6. Hold-out test set evaluation (80/20 split)")
print("=" * 70)

# Parse C value from best model name
import re
c_match = re.search(r'C=([\d.]+)', overall_best[0])
best_C = float(c_match.group(1)) if c_match else 1.0

# Determine which feature set to use
if 'combined' in overall_best[0]:
    X_final = X_combined
    feat_desc = f"combined ({X_combined.shape[1]}d)"
elif 'text' in overall_best[0]:
    X_final = X_text
    feat_desc = f"text ({X_text.shape[1]}d)"
else:
    X_final = X_struct
    feat_desc = f"struct ({X_struct.shape[1]}d)"

print(f"  Model: LogReg C={best_C}")
print(f"  Features: {feat_desc}")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

model = LogisticRegression(C=best_C, max_iter=3000)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test, y_pred_test)

print(f"\n  Training set accuracy: {train_acc:.2%}")
print(f"  Test set accuracy: {test_acc:.2%}")
print(f"  Gap: {train_acc - test_acc:.2%}")

print(f"\n  === Classification Report ===")
print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES))

print(f"  === Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred_test)
# Format and print confusion matrix
print(f"  {'':>30s} | {'Pred:Memory':>12s} | {'Pred:Starry':>12s} | {'Pred:Lily':>12s}")
print(f"  {'-'*75}")
for i, name in enumerate(['True:Memory', 'True:Starry', 'True:Lily']):
    print(f"  {name:>30s} | {cm[i][0]:>12d} | {cm[i][1]:>12d} | {cm[i][2]:>12d}")


# ============================================================
# 8. Feature Importance (Top-20)
# ============================================================
print(f"\n" + "=" * 70)
print("8. Feature importance (LogReg |coef| mean across classes, Top-20)")
print("=" * 70)

# Full training
full_model = LogisticRegression(C=best_C, max_iter=3000)
full_model.fit(X_final, y)

# Construct feature names
feature_names = list(struct_cols)
for col, tfidf in tfidf_models.items():
    feature_names += [f"tfidf_{col}::{w}" for w in tfidf.get_feature_names_out()]

importances = np.abs(full_model.coef_).mean(axis=0)  # mean across classes
top_idx = np.argsort(importances)[::-1][:20]

print(f"\n  {'Rank':>4s}  {'Feature':<45s} | {'|coef| mean':>10s}")
print(f"  {'-'*65}")
for rank, i in enumerate(top_idx, 1):
    bar = "█" * int(importances[i] / importances[top_idx[0]] * 20)
    print(f"  {rank:>4d}  {feature_names[i]:<45s} | {importances[i]:>10.4f}  {bar}")


# ============================================================
# 9. Summary
# ============================================================
print(f"\n" + "=" * 70)
print("8. Summary")
print("=" * 70)
print(f"""
  Final model:     Logistic Regression (L2, C={best_C})
  Feature dimension:     {X_final.shape[1]}  (structured {X_struct.shape[1]} + text TF-IDF {X_text.shape[1]})
  5-fold CV:    {overall_best[2]:.2%} ± {overall_best[3]:.2%}
  Hold-out:     {test_acc:.2%}
  Train-CV gap: {overall_best[4]:.2%}

  Comparison:
    vs Naive Bayes:    +{overall_best[2] - best_nb[2]:.1%} CV accuracy
    vs Decision Tree:  +{overall_best[2] - best_dt[2]:.1%} CV accuracy

  Next step:  Use exported weights to write pred.py (pure numpy inference)
""")