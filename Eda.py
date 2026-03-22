# ============================================================
#   PLACEMENT INTELLIGENCE DASHBOARD
#   Day 2 — Exploratory Data Analysis (EDA)
#   File: eda.py
#   Run: python eda.py
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
BG_COLOR     = '#0f0c29'
PANEL_COLOR  = '#1a1540'
GOLD         = '#f9ca24'
ORANGE_GOLD  = '#f0932b'
LIGHT_PURPLE = '#302b63'
PURPLE_ACC   = '#7d6fd0'
TEXT_COLOR   = '#ffffff'
GRID_COLOR   = '#2e2a5e'

mpl.rcParams.update({
    'figure.facecolor' : BG_COLOR,
    'axes.facecolor'   : PANEL_COLOR,
    'axes.edgecolor'   : LIGHT_PURPLE,
    'axes.labelcolor'  : TEXT_COLOR,
    'axes.titlecolor'  : GOLD,
    'xtick.color'      : TEXT_COLOR,
    'ytick.color'      : TEXT_COLOR,
    'text.color'       : TEXT_COLOR,
    'grid.color'       : GRID_COLOR,
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.4,
})

PALETTE = [GOLD, ORANGE_GOLD, PURPLE_ACC, LIGHT_PURPLE, '#e84393']

print("=" * 55)
print("   PLACEMENT INTELLIGENCE DASHBOARD")
print("   Day 2 — EDA Started")
print("=" * 55)

# ─────────────────────────────────────────
# LOAD CLEANED DATA
# ─────────────────────────────────────────
df = pd.read_csv('data/cleaned_data.csv')
print(f"\n✅ Cleaned data loaded — {df.shape[0]} students, {df.shape[1]} columns")

placed     = df[df['Placement'] == 'Placed']
not_placed = df[df['Placement'] == 'Not Placed']

print(f"   Placed     : {len(placed)}")
print(f"   Not Placed : {len(not_placed)}")

# ─────────────────────────────────────────
# CHART 1 — CGPA vs Placement (Box Plot)
# ─────────────────────────────────────────
print("\n📊 Chart 1: CGPA vs Placement...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)

data_to_plot = [placed['CGPA'].values, not_placed['CGPA'].values]
bp = ax.boxplot(data_to_plot,
                patch_artist=True,
                labels=['Placed', 'Not Placed'],
                widths=0.5)

bp['boxes'][0].set_facecolor(GOLD)
bp['boxes'][1].set_facecolor(ORANGE_GOLD)
for whisker in bp['whiskers']:
    whisker.set_color(TEXT_COLOR)
for cap in bp['caps']:
    cap.set_color(TEXT_COLOR)
for median in bp['medians']:
    median.set_color(BG_COLOR)
    median.set_linewidth(2)

ax.set_title('CGPA Distribution — Placed vs Not Placed',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_ylabel('CGPA', fontsize=12)
ax.set_xlabel('Placement Status', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/02_cgpa_vs_placement.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/02_cgpa_vs_placement.png")

# ─────────────────────────────────────────
# CHART 2 — IQ vs Placement (Box Plot)
# ─────────────────────────────────────────
print("\n📊 Chart 2: IQ vs Placement...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)

data_iq = [placed['IQ'].values, not_placed['IQ'].values]
bp2 = ax.boxplot(data_iq,
                 patch_artist=True,
                 labels=['Placed', 'Not Placed'],
                 widths=0.5)

bp2['boxes'][0].set_facecolor(PURPLE_ACC)
bp2['boxes'][1].set_facecolor(LIGHT_PURPLE)
for whisker in bp2['whiskers']:
    whisker.set_color(TEXT_COLOR)
for cap in bp2['caps']:
    cap.set_color(TEXT_COLOR)
for median in bp2['medians']:
    median.set_color(GOLD)
    median.set_linewidth(2)

ax.set_title('IQ Distribution — Placed vs Not Placed',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_ylabel('IQ Score', fontsize=12)
ax.set_xlabel('Placement Status', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/03_iq_vs_placement.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/03_iq_vs_placement.png")

# ─────────────────────────────────────────
# CHART 3 — Communication Skills vs CGPA (Scatter Plot)
# ─────────────────────────────────────────
print("\n📊 Chart 3: Communication Skills vs CGPA Scatter Plot...")

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG_COLOR)

# Plot placed students
ax.scatter(placed['Communication_Skills'],
           placed['CGPA'],
           color=GOLD, alpha=0.5, s=40,
           label='Placed', edgecolors='none')

# Plot not placed students
ax.scatter(not_placed['Communication_Skills'],
           not_placed['CGPA'],
           color=ORANGE_GOLD, alpha=0.4, s=40,
           label='Not Placed', edgecolors='none')

# Add trend lines
z1 = np.polyfit(placed['Communication_Skills'], placed['CGPA'], 1)
p1 = np.poly1d(z1)
x_line = np.linspace(df['Communication_Skills'].min(),
                     df['Communication_Skills'].max(), 100)
ax.plot(x_line, p1(x_line), color=GOLD,
        linewidth=2, linestyle='--', alpha=0.9)

z2 = np.polyfit(not_placed['Communication_Skills'], not_placed['CGPA'], 1)
p2 = np.poly1d(z2)
ax.plot(x_line, p2(x_line), color=ORANGE_GOLD,
        linewidth=2, linestyle='--', alpha=0.9)

ax.set_title('Communication Skills vs CGPA — Placed vs Not Placed',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Communication Skills Score', fontsize=12)
ax.set_ylabel('CGPA', fontsize=12)
ax.legend(facecolor=PANEL_COLOR, edgecolor=LIGHT_PURPLE,
          labelcolor=TEXT_COLOR, fontsize=11)
ax.grid(alpha=0.2)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/04_communication_vs_placement.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/04_communication_vs_placement.png")

# ─────────────────────────────────────────
# CHART 4 — Projects & Skills Comparison (Horizontal Bar)
# ─────────────────────────────────────────
print("\n📊 Chart 4: Projects & Skills Horizontal Bar...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG_COLOR)

# Compare multiple metrics between placed and not placed
metrics = ['CGPA', 'IQ', 'Communication_Skills',
           'Academic_Performance', 'Extra_Curricular_Score',
           'Projects_Completed']

placed_means     = [placed[m].mean() for m in metrics]
not_placed_means = [not_placed[m].mean() for m in metrics]

# Normalize values to 0-10 scale for fair comparison
def normalize(vals):
    max_val = max(vals)
    return [v / max_val * 10 for v in vals]

placed_norm     = normalize(placed_means)
not_placed_norm = normalize(not_placed_means)

y      = np.arange(len(metrics))
height = 0.35

bars1 = ax.barh(y + height/2, placed_norm, height,
                label='Placed', color=GOLD,
                edgecolor=LIGHT_PURPLE, linewidth=1.2)

bars2 = ax.barh(y - height/2, not_placed_norm, height,
                label='Not Placed', color=ORANGE_GOLD,
                edgecolor=LIGHT_PURPLE, linewidth=1.2)

ax.set_title('Student Profile Comparison — Placed vs Not Placed',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Normalized Score (out of 10)', fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels(metrics, fontsize=11)
ax.legend(facecolor=PANEL_COLOR, edgecolor=LIGHT_PURPLE,
          labelcolor=TEXT_COLOR, fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

for bar in bars1:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.1f}',
            va='center', color=GOLD, fontweight='bold', fontsize=9)
for bar in bars2:
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.1f}',
            va='center', color=ORANGE_GOLD, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('visuals/05_projects_vs_placement.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/05_projects_vs_placement.png")

# ─────────────────────────────────────────
# CHART 5 — Internship Impact on Placement (Grouped Bar)
# ─────────────────────────────────────────
print("\n📊 Chart 5: Internship Impact...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)

internship_placement = df.groupby(['Internship_Experience', 'Placement']).size().unstack()

x      = np.arange(len(internship_placement.index))
width  = 0.35

bars1 = ax.bar(x - width/2,
               internship_placement['Placed'],
               width, label='Placed',
               color=GOLD, edgecolor=LIGHT_PURPLE, linewidth=1.2)

bars2 = ax.bar(x + width/2,
               internship_placement['Not Placed'],
               width, label='Not Placed',
               color=ORANGE_GOLD, edgecolor=LIGHT_PURPLE, linewidth=1.2)

ax.set_title('Internship Experience Impact on Placement',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_ylabel('Number of Students', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(internship_placement.index)
ax.legend(facecolor=PANEL_COLOR, edgecolor=LIGHT_PURPLE,
          labelcolor=TEXT_COLOR)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20,
            str(int(bar.get_height())),
            ha='center', color=GOLD,
            fontweight='bold', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 20,
            str(int(bar.get_height())),
            ha='center', color=ORANGE_GOLD,
            fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('visuals/06_internship_impact.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/06_internship_impact.png")

# ─────────────────────────────────────────
# CHART 6 — Correlation Heatmap
# ─────────────────────────────────────────
print("\n📊 Chart 6: Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_COLOR)

corr_cols = ['IQ', 'Prev_Sem_Result', 'CGPA',
             'Academic_Performance', 'Extra_Curricular_Score',
             'Communication_Skills', 'Projects_Completed',
             'Placement_Flag']

corr_matrix = df[corr_cols].corr()

cmap = sns.diverging_palette(260, 45, s=90, l=45, as_cmap=True)

sns.heatmap(corr_matrix,
            annot=True, fmt='.2f',
            cmap=cmap,
            linewidths=0.5,
            linecolor=BG_COLOR,
            ax=ax,
            annot_kws={'size': 9, 'color': TEXT_COLOR},
            cbar_kws={'shrink': 0.8})

ax.set_title('Correlation Heatmap — All Features',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_facecolor(PANEL_COLOR)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('visuals/07_correlation_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/07_correlation_heatmap.png")

# ─────────────────────────────────────────
# CHART 7 — CGPA vs Placement Rate (Area Chart)
# ─────────────────────────────────────────
print("\n📊 Chart 7: CGPA vs Placement Rate Area Chart...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)

# Create CGPA bins and calculate placement rate per bin
df['CGPA_Bin'] = pd.cut(df['CGPA'], bins=10)
area_data = df.groupby('CGPA_Bin')['Placement_Flag'].mean() * 100
area_data = area_data.dropna()

x_labels = [str(interval.mid.round(1)) for interval in area_data.index]
x_pos    = np.arange(len(x_labels))

# Area chart
ax.fill_between(x_pos, area_data.values,
                color=GOLD, alpha=0.4)
ax.plot(x_pos, area_data.values,
        color=GOLD, linewidth=2.5,
        marker='o', markersize=7,
        markerfacecolor=ORANGE_GOLD,
        markeredgecolor=GOLD)

# Annotate each point
for i, val in enumerate(area_data.values):
    ax.text(i, val + 1.5, f'{val:.1f}%',
            ha='center', color=GOLD,
            fontweight='bold', fontsize=9)

ax.set_title('Placement Rate (%) Across CGPA Range',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('CGPA (Midpoint)', fontsize=12)
ax.set_ylabel('Placement Rate (%)', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=30, ha='right')
ax.set_ylim(0, 100)
ax.grid(alpha=0.2)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/08_cgpa_band_placement_rate.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/08_cgpa_band_placement_rate.png")

# ─────────────────────────────────────────
# CHART 8 — IQ Band Placement Rate (Horizontal Bar)
# ─────────────────────────────────────────
print("\n📊 Chart 8: IQ Band Placement Rate Horizontal Bar...")

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_COLOR)

iq_placement = df.groupby('IQ_Band')['Placement_Flag'].mean() * 100
iq_placement = iq_placement.sort_values(ascending=True)

colors = [GOLD, ORANGE_GOLD, PURPLE_ACC]
bars   = ax.barh(iq_placement.index,
                 iq_placement.values,
                 color=colors[:len(iq_placement)],
                 edgecolor=LIGHT_PURPLE,
                 linewidth=1.2, height=0.4)

ax.set_title('Placement Rate (%) by IQ Band',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Placement Rate (%)', fontsize=12)
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

for bar, val in zip(bars, iq_placement.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%',
            va='center', fontweight='bold',
            color=GOLD, fontsize=12)

plt.tight_layout()
plt.savefig('visuals/09_iq_band_placement_rate.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/09_iq_band_placement_rate.png")

# ─────────────────────────────────────────
# CHART 9 — Overall Score Distribution (Histogram)
# ─────────────────────────────────────────
print("\n📊 Chart 9: Overall Score Distribution...")

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_COLOR)

ax.hist(placed['Overall_Score'],
        bins=30, alpha=0.8,
        color=GOLD, edgecolor=BG_COLOR,
        label='Placed')

ax.hist(not_placed['Overall_Score'],
        bins=30, alpha=0.6,
        color=ORANGE_GOLD, edgecolor=BG_COLOR,
        label='Not Placed')

ax.set_title('Overall Score Distribution — Placed vs Not Placed',
             fontsize=14, fontweight='bold', color=GOLD, pad=15)
ax.set_xlabel('Overall Score', fontsize=12)
ax.set_ylabel('Number of Students', fontsize=12)
ax.legend(facecolor=PANEL_COLOR, edgecolor=LIGHT_PURPLE,
          labelcolor=TEXT_COLOR)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(PANEL_COLOR)

plt.tight_layout()
plt.savefig('visuals/10_overall_score_distribution.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/10_overall_score_distribution.png")

# ─────────────────────────────────────────
# CHART 10 — Key Insights Summary (4-in-1)
# ─────────────────────────────────────────
print("\n📊 Chart 10: Key Insights Summary...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('✦ Key Placement Insights Summary ✦',
             fontsize=18, fontweight='bold', color=GOLD, y=1.01)

# ── Mini Chart A — Avg CGPA ──
metrics     = ['Placed', 'Not Placed']
cgpa_vals   = [placed['CGPA'].mean(), not_placed['CGPA'].mean()]
b1 = axes[0,0].bar(metrics, cgpa_vals,
                   color=[GOLD, ORANGE_GOLD],
                   edgecolor=LIGHT_PURPLE, width=0.4)
axes[0,0].set_title('Average CGPA', fontweight='bold', color=GOLD)
axes[0,0].set_ylabel('CGPA')
axes[0,0].set_ylim(0, 10)
axes[0,0].grid(axis='y', alpha=0.3)
axes[0,0].set_facecolor(PANEL_COLOR)
for bar, val in zip(b1, cgpa_vals):
    axes[0,0].text(bar.get_x() + bar.get_width()/2,
                   val + 0.1, f'{val:.2f}',
                   ha='center', color=GOLD, fontweight='bold')

# ── Mini Chart B — Avg IQ ──
iq_vals = [placed['IQ'].mean(), not_placed['IQ'].mean()]
b2 = axes[0,1].bar(metrics, iq_vals,
                   color=[PURPLE_ACC, LIGHT_PURPLE],
                   edgecolor=GOLD, width=0.4)
axes[0,1].set_title('Average IQ Score', fontweight='bold', color=GOLD)
axes[0,1].set_ylabel('IQ Score')
axes[0,1].set_ylim(0, max(iq_vals) + 20)
axes[0,1].grid(axis='y', alpha=0.3)
axes[0,1].set_facecolor(PANEL_COLOR)
for bar, val in zip(b2, iq_vals):
    axes[0,1].text(bar.get_x() + bar.get_width()/2,
                   val + 0.5, f'{val:.1f}',
                   ha='center', color=GOLD, fontweight='bold')

# ── Mini Chart C — Avg Extra Curricular ──
ec_vals = [placed['Extra_Curricular_Score'].mean(),
           not_placed['Extra_Curricular_Score'].mean()]
b3 = axes[1,0].bar(metrics, ec_vals,
                   color=[GOLD, ORANGE_GOLD],
                   edgecolor=LIGHT_PURPLE, width=0.4)
axes[1,0].set_title('Avg Extra Curricular Score', fontweight='bold', color=GOLD)
axes[1,0].set_ylabel('Score')
axes[1,0].set_ylim(0, max(ec_vals) + 2)
axes[1,0].grid(axis='y', alpha=0.3)
axes[1,0].set_facecolor(PANEL_COLOR)
for bar, val in zip(b3, ec_vals):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   val + 0.1, f'{val:.2f}',
                   ha='center', color=GOLD, fontweight='bold')

# ── Mini Chart D — Avg Projects ──
pr_vals = [placed['Projects_Completed'].mean(),
           not_placed['Projects_Completed'].mean()]
b4 = axes[1,1].bar(metrics, pr_vals,
                   color=[PURPLE_ACC, LIGHT_PURPLE],
                   edgecolor=GOLD, width=0.4)
axes[1,1].set_title('Avg Projects Completed', fontweight='bold', color=GOLD)
axes[1,1].set_ylabel('Projects')
axes[1,1].set_ylim(0, max(pr_vals) + 1)
axes[1,1].grid(axis='y', alpha=0.3)
axes[1,1].set_facecolor(PANEL_COLOR)
for bar, val in zip(b4, pr_vals):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   val + 0.05, f'{val:.2f}',
                   ha='center', color=GOLD, fontweight='bold')

plt.tight_layout()
plt.savefig('visuals/11_key_insights_summary.png',
            dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.show()
print("✅ Saved → visuals/11_key_insights_summary.png")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("   ✅ DAY 2 COMPLETE — EDA DONE!")
print("=" * 55)
print(f"""
   📊 10 Charts Generated & Saved:
   ├── 02_cgpa_vs_placement.png
   ├── 03_iq_vs_placement.png
   ├── 04_communication_vs_placement.png
   ├── 05_projects_vs_placement.png
   ├── 06_internship_impact.png
   ├── 07_correlation_heatmap.png
   ├── 08_cgpa_band_placement_rate.png
   ├── 09_iq_band_placement_rate.png
   ├── 10_overall_score_distribution.png
   └── 11_key_insights_summary.png

   🔑 Key Findings:
   • Placed students avg CGPA  : {placed['CGPA'].mean():.2f}
   • Not Placed avg CGPA       : {not_placed['CGPA'].mean():.2f}
   • Placed students avg IQ    : {placed['IQ'].mean():.1f}
   • Not Placed avg IQ         : {not_placed['IQ'].mean():.1f}
   • Placed avg projects       : {placed['Projects_Completed'].mean():.2f}
   • Not Placed avg projects   : {not_placed['Projects_Completed'].mean():.2f}

   Next: Load cleaned_data.csv into Power BI!
""")