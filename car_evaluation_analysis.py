# -*- coding: utf-8 -*-
# Car Evaluation - simple data analysis
# Dataset: UCI Car Evaluation (data/car.data)
# Goal: see what makes a car "acceptable" (target class: unacc / acc / good / vgood).

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


# Work next to this script, not in whatever folder you ran python from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)


# 1. Load the data
# The file has no header, so we give the column names ourselves.
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", names=column_names)

# The target has 4 values. Make a simple 0/1 column "acceptable":
# anything that is not "unacc" counts as an acceptable car.
df["acceptable"] = (df["class"] != "unacc").astype(int)

print("Shape:", df.shape)
print(df.head())


# 2. Quick look at the data
print("\nMissing values (NaN):")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nValues in each column:")
for col in column_names:
    print(f"  {col}: {dict(df[col].value_counts())}")

# Data-quality table for Excel: per column, NaN count and how many unique values.
quality = pd.DataFrame({
    "column": df.columns,
    "missing_nan": df.isna().sum().to_numpy(),
    "n_unique": [df[c].nunique() for c in df.columns],
})


# 3. Target: how are the classes spread out?
print("\nClass distribution:")
target_counts = df["class"].value_counts()
print(target_counts)

base_rate = df["acceptable"].mean()
print(f"\nAcceptable cars (acc/good/vgood): {base_rate:.1%}")

# Table for Excel: class, count, percent.
target_table = target_counts.rename_axis("class").reset_index(name="count")
target_table["percent"] = (target_table["count"] / len(df) * 100).round(1)

plt.figure(figsize=(5, 4))
plt.bar(target_counts.index, target_counts.values, color="steelblue")
plt.title("Car class distribution")
plt.ylabel("cars")
plt.tight_layout()
plt.savefig("figures/01_target.png")
plt.close()


# 4. Features vs target: acceptance rate per category
# Every feature is categorical, so for each one we look at the share of
# acceptable cars inside each group, plus the "lift" vs the overall rate.
features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# A natural order for the categories, so plots read left-to-right "worse -> better".
nice_order = {
    "buying": ["low", "med", "high", "vhigh"],
    "maint": ["low", "med", "high", "vhigh"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}

cat_tables = {}  # keep each table for the Excel report

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), features):
    group = df.groupby(col)["acceptable"].agg(["mean", "size"])
    group.columns = ["rate", "n"]
    group = group.reindex(nice_order[col])          # put categories in a sensible order
    group["rate_%"] = (group["rate"] * 100).round(1)
    group["lift"] = (group["rate"] / base_rate).round(2)

    print(f"\nAcceptance rate by {col}:")
    print(group[["rate_%", "n", "lift"]])

    # reset_index() makes the category a real column (good Excel table, not one column).
    cat_tables[col] = group[["rate_%", "n", "lift"]].reset_index()

    ax.bar(group.index.astype(str), group["rate"] * 100, color="seagreen")
    ax.axhline(base_rate * 100, ls="--", color="gray")
    ax.set_title(col)
    ax.set_ylabel("acceptable, %")
fig.suptitle("Acceptance rate by feature (dashed = overall rate)", fontweight="bold")
fig.tight_layout()
fig.savefig("figures/02_acceptance_by_feature.png")
plt.close(fig)


# 5. Model: predict the car class (4 classes)
# All features are text, so we turn them into numbers with get_dummies (one-hot).
X = df.drop(columns=["class", "acceptable"])
y = df["class"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n=== Random forest report ===")
print(classification_report(y_test, pred))

# Same report as a table for Excel.
report_dict = classification_report(y_test, pred, output_dict=True)
report_table = pd.DataFrame(report_dict).T.round(3).reset_index()
report_table = report_table.rename(columns={"index": "class"})


# 6. Is the model good? Cross-validation + compare with other models
# One split can get lucky, so we average over 5 shuffled folds.
print("\n=== Cross-validation (accuracy, 5 folds) ===")
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "dummy (baseline)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=2000),
    "random forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

cv_results = []
for name in models:
    acc = cross_val_score(models[name], X, y, cv=folds, scoring="accuracy")
    f1 = cross_val_score(models[name], X, y, cv=folds, scoring="f1_macro")
    print(f"{name:22} accuracy = {acc.mean():.3f}   f1_macro = {f1.mean():.3f}")
    cv_results.append({"model": name,
                       "accuracy": round(acc.mean(), 3),
                       "f1_macro": round(f1.mean(), 3)})

cv_results = pd.DataFrame(cv_results)


# 7. Which features matter most? (random forest importance)
# Each feature became several one-hot columns; we add their importances back up
# so we get one number per original feature.
importance_per_dummy = pd.Series(model.feature_importances_, index=X.columns)
importance = {}
for col in features:
    importance[col] = importance_per_dummy[[c for c in X.columns if c.startswith(col + "_")]].sum()
importance = pd.Series(importance).sort_values(ascending=False).round(3)
importance_table = importance.reset_index()
importance_table.columns = ["feature", "importance"]

print("\nFeature importance:")
print(importance_table.to_string(index=False))

plt.figure(figsize=(6, 4))
plt.barh(importance.index[::-1], importance.values[::-1], color="darkorange")
plt.title("Feature importance (random forest)")
plt.tight_layout()
plt.savefig("figures/03_feature_importance.png")
plt.close()


# 8. Confusion matrix: where does the model make mistakes?
# Rows = the real class, columns = what the model predicted.
labels = ["unacc", "acc", "good", "vgood"]
confusion = pd.crosstab(y_test, pred).reindex(index=labels, columns=labels, fill_value=0)

plt.figure(figsize=(6, 5))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix (rows = real, cols = predicted)")
plt.ylabel("real class")
plt.xlabel("predicted class")
plt.tight_layout()
plt.savefig("figures/04_confusion_matrix.png")
plt.close()

confusion_table = confusion.reset_index().rename(columns={"index": "real_class"})


# 9. Save everything to one Excel file, each table on its own sheet (page).
with pd.ExcelWriter("results.xlsx") as writer:
    quality.to_excel(writer, sheet_name="data_quality", index=False)
    target_table.to_excel(writer, sheet_name="target_distribution", index=False)
    cv_results.to_excel(writer, sheet_name="model_comparison", index=False)
    report_table.to_excel(writer, sheet_name="model_report", index=False)
    importance_table.to_excel(writer, sheet_name="feature_importance", index=False)
    confusion_table.to_excel(writer, sheet_name="confusion_matrix", index=False)
    for col in cat_tables:
        cat_tables[col].to_excel(writer, sheet_name=f"by_{col}"[:31], index=False)

print("\nResults saved to results.xlsx (one sheet per table).")
print("Done. Figures saved in the figures/ folder.")
