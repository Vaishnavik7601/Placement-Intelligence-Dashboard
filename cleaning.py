# ============================================================
#   PLACEMENT INTELLIGENCE DASHBOARD
#   Day 1 — Data Cleaning
#   File: cleaning.py
#   Run: python cleaning.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# THEME — Midnight Purple + Gold
# ─────────────────────────────────────────
BG_COLOR      = '#0f0c29'   # Deep midnight background
PANEL_COLOR   = '#1a1540'   # Slightly lighter panel
GOLD          = '#f9ca24'   # Primary gold
ORANGE_GOLD   = '#f0932b'   # Secondary gold/orange
LIGHT_PURPLE  = '#302b63'   # Accent purple
TEXT_COLOR    = '#ffffff'   # White text
GRID_COLOR    = '#2e2a5e'   # Subtle grid lines

# Apply global matplotlib theme
mpl.rcParams.update({
    'figure.facecolor'  : BG_COLOR,
    'axes.facecolor'    : PANEL_COLOR,
    'axes.edgecolor'    : LIGHT_PURPLE,
    'axes.labelcolor'   : TEXT_COLOR,
    'axes.titlecolor'   : GOLD,
    'xtick.color'       : TEXT_COLOR,
    'ytick.color'       : TEXT_COLOR,
    'text.color'        : TEXT_COLOR,
    'grid.color'        : GRID_COLOR,
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.5,
})

print("=" * 50)
print("   PLACEMENT INTELLIGENCE DASHBOARD")
print("   Day 1 — Data Cleaning Started")
print("=" * 50)

# ─────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────
print("\n STEP 1: Loading dataset...")

df = pd.read_csv('data/rawdata.csv')

print(f" Dataset loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"   Columns : {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 2 — Drop Unnecessary Columns
# ─────────────────────────────────────────
print("\n STEP 2: Dropping unnecessary columns...")

df.drop(columns=['College_ID'], inplace=True)

print(" College_ID dropped!")
print(f"   Remaining columns: {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 3 — Check & Handle Missing Values
# ─────────────────────────────────────────
print("\n STEP 3: Checking missing values...")

missing = df.isnull().sum()
print(f"   Missing values per column:\n{missing}")

num_cols = ['IQ', 'Prev_Sem_Result', 'CGPA',
            'Academic_Performance', 'Extra_Curricular_Score',
            'Communication_Skills', 'Projects_Completed']

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f" {col} → filled with median")

cat_cols = ['Internship_Experience', 'Placement']
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f" {col} → filled with mode")

print(f" Missing values handled! Remaining nulls: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# STEP 4 — Remove Duplicates
# ─────────────────────────────────────────
print("\n STEP 4: Removing duplicates...")

before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]

print(f" Duplicates removed: {before - after}")
print(f"   Rows remaining: {after}")

# ─────────────────────────────────────────
# STEP 5 — Standardize Values
# ─────────────────────────────────────────
print("\n STEP 5: Standardizing values...")

print("Raw Placement values:", df['Placement'].unique())

df['Placement'] = df['Placement'].astype(str).str.strip()
df['Placement'] = df['Placement'].map({
    'Yes': 'Placed',
    'No': 'Not Placed',
    'Placed': 'Placed',
    'Not Placed': 'Not Placed'
})
print("Fixed Placement values:", df['Placement'].unique())

df['Internship_Experience'] = df['Internship_Experience'].astype(str).str.strip().str.title()
print(f"   Internship unique values: {df['Internship_Experience'].unique()}")

print(" Values standardized!")

# ─────────────────────────────────────────
# STEP 6 — Add New Useful Columns
# ─────────────────────────────────────────
print("\n STEP 6: Adding new columns...")

# 1. Placement Flag
df['Placement_Flag'] = df['Placement'].apply(
    lambda x: 1 if x == 'Placed' else 0
)
print(" Placement_Flag added")

# 2. CGPA Band
def cgpa_band(cgpa):
    if cgpa >= 8.5:
        return 'Excellent (8.5+)'
    elif cgpa >= 7.0:
        return 'Good (7.0-8.5)'
    elif cgpa >= 5.5:
        return 'Average (5.5-7.0)'
    else:
        return 'Below Average (<5.5)'

df['CGPA_Band'] = df['CGPA'].apply(cgpa_band)
print(" CGPA_Band added")

# 3. Overall Score
df['Overall_Score'] = round(
    (df['CGPA']                   * 0.30) +
    (df['Communication_Skills']   * 0.20) +
    (df['Academic_Performance']   * 0.20) +
    (df['Extra_Curricular_Score'] * 0.15) +
    (df['Projects_Completed']     * 0.15), 2
)
print(" Overall_Score added")

# 4. IQ Band
def iq_band(iq):
    if iq >= 120:
        return 'High (120+)'
    elif iq >= 100:
        return 'Average (100-120)'
    else:
        return 'Below Average (<100)'

df['IQ_Band'] = df['IQ'].apply(iq_band)
print(" IQ_Band added")

# ─────────────────────────────────────────
# STEP 7 — Final Summary
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("   FINAL CLEANED DATASET SUMMARY")
print("=" * 50)
print(f"   Total Students   : {df.shape[0]}")
print(f"   Total Columns    : {df.shape[1]}")
print(f"   Placed           : {df[df['Placement'] == 'Placed'].shape[0]}")
print(f"   Not Placed       : {df[df['Placement'] == 'Not Placed'].shape[0]}")
print(f"   Placement Rate   : {round(df['Placement_Flag'].mean() * 100, 2)}%")
print(f"   Avg CGPA         : {round(df['CGPA'].mean(), 2)}")
print(f"   Avg IQ           : {round(df['IQ'].mean(), 2)}")
print(f"   Avg Projects     : {round(df['Projects_Completed'].mean(), 2)}")
print(f"\n   Final Columns    : {df.columns.tolist()}")

# ─────────────────────────────────────────
# STEP 8 — Export Cleaned Data
# ─────────────────────────────────────────
print("\n STEP 8: Saving cleaned data...")

df.to_csv('data/cleaned_data.csv', index=False)
print("✅ Saved → data/cleaned_data.csv")

# ─────────────────────────────────────────
# STEP 9 — Overview Charts (Midnight Purple + Gold)
# ─────────────────────────────────────────
print("\n STEP 9: Generating overview charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_COLOR)

# Main title
fig.suptitle('✦ Placement Intelligence Dashboard ',
             fontsize=20, fontweight='bold',
             color=GOLD, y=1.02)

# ── Chart 1 — Placed vs Not Placed ──
placement_counts = df['Placement'].value_counts()
bar_colors = [GOLD if x == 'Placed' else ORANGE_GOLD
              for x in placement_counts.index]

bars = axes[0].bar(placement_counts.index,
                   placement_counts.values,
                   color=bar_colors, edgecolor=LIGHT_PURPLE,
                   linewidth=1.5, width=0.5)

axes[0].set_title('Placed vs Not Placed', fontweight='bold',
                  fontsize=13, color=GOLD, pad=12)
axes[0].set_ylabel('Number of Students', color=TEXT_COLOR)
axes[0].set_ylim(0, df.shape[0] + 500)
axes[0].yaxis.set_major_locator(plt.MultipleLocator(2000))
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_facecolor(PANEL_COLOR)

for bar, val in zip(bars, placement_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 val + 150, str(val),
                 ha='center', fontweight='bold',
                 color=GOLD, fontsize=12)

# ── Chart 2 — CGPA Band Pie ──
cgpa_counts = df['CGPA_Band'].value_counts()
pie_colors  = [GOLD, ORANGE_GOLD, LIGHT_PURPLE, '#7d6fd0']
wedges, texts, autotexts = axes[1].pie(
    cgpa_counts.values,
    labels=cgpa_counts.index,
    autopct='%1.1f%%',
    colors=pie_colors,
    startangle=90,
    textprops={'fontsize': 9, 'color': TEXT_COLOR},
    wedgeprops={'edgecolor': BG_COLOR, 'linewidth': 2}
)
for autotext in autotexts:
    autotext.set_color(BG_COLOR)
    autotext.set_fontweight('bold')

axes[1].set_title('CGPA Band Distribution', fontweight='bold',
                  fontsize=13, color=GOLD, pad=12)
axes[1].set_facecolor(BG_COLOR)

# ── Chart 3 — Internship Experience ──
intern_counts = df['Internship_Experience'].value_counts()
intern_colors = [GOLD if x == 'Yes' else ORANGE_GOLD
                 for x in intern_counts.index]

bars3 = axes[2].bar(intern_counts.index,
                    intern_counts.values,
                    color=intern_colors,
                    edgecolor=LIGHT_PURPLE,
                    linewidth=1.5, width=0.5)

axes[2].set_title('Internship Experience', fontweight='bold',
                  fontsize=13, color=GOLD, pad=12)
axes[2].set_ylabel('Number of Students', color=TEXT_COLOR)
axes[2].set_ylim(0, df.shape[0] + 500)
axes[2].yaxis.set_major_locator(plt.MultipleLocator(2000))
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_facecolor(PANEL_COLOR)

for bar, val in zip(bars3, intern_counts.values):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 val + 150, str(val),
                 ha='center', fontweight='bold',
                 color=GOLD, fontsize=12)

plt.tight_layout()
plt.savefig('visuals/01_overview.png',
            dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.show()
print("✅ Chart saved → visuals/01_overview.png")

# ─────────────────────────────────────────
# DONE!
# ─────────────────────────────────────────
