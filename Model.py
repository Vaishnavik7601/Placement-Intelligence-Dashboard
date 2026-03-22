# ============================================================
#   PLACEMENT INTELLIGENCE DASHBOARD
#   Day 4 — Machine Learning Model
#   File: model.py
#   Run: python model.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# THEME — Midnight Purple + Gold
# ─────────────────────────────────────────
BG_COLOR     = '#0f0c29'
PANEL_COLOR  = '#1a1540'
GOLD         = '#f9ca24'
ORANGE_GOLD  = '#f0932b'
LIGHT_PURPLE = '#302b63'
PURPLE_ACC   = '#7d6fd0'
TEXT_COLOR   = '#ffffff'

mpl.rcParams.update({
    'figure.facecolor' : BG_COLOR,
    'axes.facecolor'   : PANEL_COLOR,
    'axes.edgecolor'   : LIGHT_PURPLE,
    'axes.labelcolor'  : TEXT_COLOR,
    'axes.titlecolor'  : GOLD,
    'xtick.color'      : TEXT_COLOR,
    'ytick.color'      : TEXT_COLOR,
    'text.color'       : TEXT_COLOR,
    'grid.color'       : '#2e2a5e',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.4,
})

print("=" * 55)
print("   PLACEMENT INTELLIGENCE DASHBOARD")
print("   ML Placement Predictor")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Load Cleaned Data
# ─────────────────────────────────────────
print("\n📂 STEP 1: Loading cleaned data...")

df = pd.read_csv('data/cleaned_data.csv')
print(f"✅ Data loaded — {df.shape[0]} students, {df.shape[1]} columns")

# ─────────────────────────────────────────
# STEP 2 — Prepare Features
# ─────────────────────────────────────────
print("\n⚙️  STEP 2: Preparing features...")

# Features (X) — what we use to predict
feature_cols = [
    'IQ',
    'Prev_Sem_Result',
    'CGPA',
    'Academic_Performance',
    'Extra_Curricular_Score',
    'Communication_Skills',
    'Projects_Completed'
]

# Convert Internship_Experience to 0/1
df['Internship_Flag'] = df['Internship_Experience'].apply(
    lambda x: 1 if str(x).strip().title() == 'Yes' else 0
)
feature_cols.append('Internship_Flag')

# Target (y) — what we want to predict
X = df[feature_cols]
y = df['Placement_Flag']

print(f"✅ Features selected: {feature_cols}")
print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")
print(f"✅ Placed: {y.sum()} | Not Placed: {(y==0).sum()}")

# ─────────────────────────────────────────
# STEP 3 — Split Data
# ─────────────────────────────────────────
print("\n✂️  STEP 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42,    # reproducible results
    stratify=y          # maintain class balance
)

print(f"✅ Training set : {X_train.shape[0]} students")
print(f"✅ Testing set  : {X_test.shape[0]} students")

# ─────────────────────────────────────────
# STEP 4 — Train Models
# ─────────────────────────────────────────
print("\n🤖 STEP 4: Training ML models...")

# Model 1 — Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred) * 100
print(f"✅ Logistic Regression Accuracy : {lr_acc:.2f}%")

# Model 2 — Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred) * 100
print(f"✅ Random Forest Accuracy       : {rf_acc:.2f}%")

# Best model
best_model = rf_model if rf_acc >= lr_acc else lr_model
best_pred  = rf_pred  if rf_acc >= lr_acc else lr_pred
best_name  = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
best_acc   = max(rf_acc, lr_acc)

print(f"\n🏆 Best Model: {best_name} ({best_acc:.2f}% accuracy)")

# ─────────────────────────────────────────
# STEP 5 — Model Evaluation
# ─────────────────────────────────────────
print("\n📊 STEP 5: Evaluating model...")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, best_pred,
      target_names=['Not Placed', 'Placed']))

# ─────────────────────────────────────────
# STEP 6 — Confusion Matrix Chart
# ─────────────────────────────────────────
print("\n📊 STEP 6: Generating Confusion Matrix...")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG_COLOR)

cm = confusion_matrix(y_test, best_pred)
cmap = sns.diverging_palette(260, 45, s=90, l=45, as_cmap=True)

sns.heatmap(cm,
            annot=True, fmt='d',
            cmap=cmap,
            linewidths=2,
            linecolor=BG_COLOR,
            ax=ax,
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'],
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_title(f'Confusion Matrix — {best_name}\nAccuracy: {best_acc:.2f}%',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/12_confusion_matrix.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/12_confusion_matrix.png")

# ─────────────────────────────────────────
# STEP 7 — Feature Importance Chart
# ─────────────────────────────────────────
print("\n📊 STEP 7: Feature Importance Chart...")

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG_COLOR)

importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]
features    = [feature_cols[i] for i in indices]
values      = importances[indices]

colors = [GOLD if v == max(values) else ORANGE_GOLD
          if v >= np.median(values) else PURPLE_ACC
          for v in values]

bars = ax.barh(features[::-1], values[::-1],
               color=colors[::-1],
               edgecolor=LIGHT_PURPLE,
               linewidth=1.2)

ax.set_title('Feature Importance — What Predicts Placement?',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

for bar, val in zip(bars, values[::-1]):
    ax.text(val + 0.002,
            bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            va='center', color=GOLD,
            fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('visuals/13_feature_importance.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/13_feature_importance.png")

# ─────────────────────────────────────────
# STEP 8 — Model Comparison Chart
# ─────────────────────────────────────────
print("\n📊 STEP 8: Model Comparison Chart...")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG_COLOR)

models      = ['Logistic Regression', 'Random Forest']
accuracies  = [lr_acc, rf_acc]
bar_colors  = [ORANGE_GOLD, GOLD]

bars = ax.bar(models, accuracies,
              color=bar_colors,
              edgecolor=LIGHT_PURPLE,
              linewidth=1.5, width=0.4)

ax.set_title('Model Accuracy Comparison',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 1, f'{val:.2f}%',
            ha='center', fontweight='bold',
            color=GOLD, fontsize=13)

plt.tight_layout()
plt.savefig('visuals/14_model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/14_model_comparison.png")

# ─────────────────────────────────────────
# STEP 9 — Live Placement Predictor
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("   🎯 LIVE PLACEMENT PREDICTOR")
print("=" * 55)
print("Enter student details to predict placement:\n")

try:
    iq           = float(input("   IQ Score (50-150)          : "))
    prev_sem     = float(input("   Previous Sem Result (0-100): "))
    cgpa         = float(input("   CGPA (0-10)                : "))
    academic     = float(input("   Academic Performance (0-10): "))
    extra        = float(input("   Extra Curricular (0-10)    : "))
    communication= float(input("   Communication Skills (0-10): "))
    projects     = float(input("   Projects Completed (0-10)  : "))
    internship   = input(      "   Internship Experience (Yes/No): ").strip().title()
    internship_f = 1 if internship == 'Yes' else 0

    student = pd.DataFrame([[
        iq, prev_sem, cgpa, academic,
        extra, communication, projects, internship_f
    ]], columns=feature_cols)

    prediction   = best_model.predict(student)[0]
    probability  = best_model.predict_proba(student)[0]
    confidence   = max(probability) * 100

    print("\n" + "=" * 55)
    if prediction == 1:
        print(f"   ✅ PREDICTION : PLACED")
    else:
        print(f"   ❌ PREDICTION : NOT PLACED")
    print(f"   🎯 CONFIDENCE : {confidence:.1f}%")
    print(f"   🤖 MODEL USED : {best_name}")
    print("=" * 55)

except Exception as e:
    print(f"\n⚠️ Predictor skipped: {e}")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print(f"""
{"=" * 55}
   ✅ ML MODEL COMPLETE!
{"=" * 55}

   🤖 Models Trained:
   ├── Logistic Regression : {lr_acc:.2f}%
   └── Random Forest       : {rf_acc:.2f}%

   🏆 Best Model : {best_name}
   🎯 Accuracy   : {best_acc:.2f}%

   📊 Charts Saved:
   ├── 12_confusion_matrix.png
   ├── 13_feature_importance.png
   └── 14_model_comparison.png

  
{"=" * 55}
""")