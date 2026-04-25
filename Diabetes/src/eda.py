from diabetes import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = load_data()

OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates the folder if it doesn't exist

# ── Clean up readmitted column ──────────────────────────────────────────────
df['readmitted_30day'] = df['readmitted_30day'].astype(int)

# ── 1. Numeric feature correlations with target ──────────────────────────────
numeric_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

correlations = df[numeric_cols + ['readmitted_30day']].corr()['readmitted_30day'].drop('readmitted_30day').sort_values()

plt.figure(figsize=(8, 5))
correlations.plot(kind='barh', color=['#d9534f' if x > 0 else '#5bc0de' for x in correlations])
plt.title('Numeric feature correlation with 30-day readmission')
plt.xlabel('Pearson correlation')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation.png', dpi=150)
plt.show()

# ── 2. Readmission rate by age group ────────────────────────────────────────
age_rate = df.groupby('age')['readmitted_30day'].mean().sort_index()

plt.figure(figsize=(8, 4))
age_rate.plot(kind='bar', color='steelblue')
plt.title('30-day readmission rate by age group')
plt.ylabel('Readmission rate')
plt.xlabel('Age bracket')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/age_readmission.png', dpi=150)
plt.show()

# ── 3. Readmission rate by number of inpatient visits ───────────────────────
inpatient_rate = df.groupby('number_inpatient')['readmitted_30day'].agg(['mean', 'count']).reset_index()
inpatient_rate = inpatient_rate[inpatient_rate['count'] > 50]  # filter low-volume

plt.figure(figsize=(8, 4))
plt.bar(inpatient_rate['number_inpatient'], inpatient_rate['mean'], color='coral')
plt.title('30-day readmission rate by prior inpatient visits')
plt.xlabel('Number of prior inpatient visits')
plt.ylabel('Readmission rate')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/inpatient_readmission.png', dpi=150)
plt.show()

# ── 4. Insulin treatment breakdown ──────────────────────────────────────────
insulin_rate = df.groupby('insulin')['readmitted_30day'].agg(['mean', 'count']).reset_index()
insulin_rate.columns = ['insulin', 'readmit_rate', 'count']
insulin_rate = insulin_rate.sort_values('readmit_rate', ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(data=insulin_rate, x='insulin', y='readmit_rate', palette='OrRd')
plt.title('30-day readmission rate by insulin status')
plt.ylabel('Readmission rate')
plt.xlabel('Insulin')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/insulin_readmission.png', dpi=150)
plt.show()

# ── 5. Chi-square tests for categorical predictors ──────────────────────────
from scipy.stats import chi2_contingency

cat_cols = ['age', 'race', 'gender', 'insulin', 'change', 'diabetesMed', 'A1Cresult']
results = []

for col in cat_cols:
    contingency = pd.crosstab(df[col].fillna('Unknown'), df['readmitted_30day'])
    chi2, p, dof, _ = chi2_contingency(contingency)
    results.append({'feature': col, 'chi2': round(chi2, 2), 'p_value': round(p, 5)})

chi2_df = pd.DataFrame(results).sort_values('chi2', ascending=False)

# ── Chi-square results visualization ────────────────────────────────────────
chi2_df_sorted = chi2_df.sort_values('chi2', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d9534f' if p < 0.05 else '#aaaaaa' for p in chi2_df_sorted['p_value']]
bars = ax.barh(chi2_df_sorted['feature'], chi2_df_sorted['chi2'], color=colors)

# Annotate p-values on each bar
for bar, (_, row) in zip(bars, chi2_df_sorted.iterrows()):
    label = f"p < 0.001" if row['p_value'] < 0.001 else f"p = {row['p_value']:.3f}"
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            label, va='center', fontsize=9,
            color='#333333')

ax.set_xlabel('Chi-square statistic')
ax.set_title('Categorical feature association with 30-day readmission')
ax.axvline(x=0, color='black', linewidth=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d9534f', label='Significant (p < 0.05)'),
                   Patch(facecolor='#aaaaaa', label='Not significant')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chi2_results.png', dpi=150)
plt.show()