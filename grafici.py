import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('matlab_simulacioni_podaci.csv')
df = df.drop(columns=['T'])

BOJA_0 = '#4e9af1'  # plava — zdrav rad
BOJA_1 = '#e05c5c'  # crvena — kvar

df['Klasa_naziv'] = df['Klasa'].map({0: 'Zdrav rad', 1: 'Gubitak faze'})
features = [c for c in df.columns if c not in ['Klasa', 'Klasa_naziv']]

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 130

# ── Grafik 1: Boxplot svih karakteristika ─────────────────────────────────
n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i, feat in enumerate(features):
    sns.boxplot(
        data=df, x='Klasa_naziv', y=feat,
        hue='Klasa_naziv',
        palette={'Zdrav rad': BOJA_0, 'Gubitak faze': BOJA_1},
        ax=axes[i], width=0.5, linewidth=1.5, legend=False
    )
    axes[i].set_title(feat, fontweight='bold')
    axes[i].set_xlabel('')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Distribucija karakteristika: Zdrav rad vs. Gubitak faze',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_boxplots.png', bbox_inches='tight')
plt.show()
print("Sačuvano: eda_boxplots.png")

# ── Grafik 2: Scatter I_s vs Brzina ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for klasa, group in df.groupby('Klasa'):
    ax.scatter(
        group['I_s'], group['Brzina'],
        label={0: 'Zdrav rad', 1: 'Gubitak faze'}[klasa],
        alpha=0.8, s=70,
        color=BOJA_0 if klasa == 0 else BOJA_1,
        edgecolors='white', linewidths=0.5
    )
ax.set_xlabel('Struja statora I_s [A]', fontsize=12)
ax.set_ylabel('Brzina rotora [rpm]', fontsize=12)
ax.set_title('Separacija klasa: I_s vs. Brzina', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('eda_scatter_Is_Brzina.png', bbox_inches='tight')
plt.show()
print("Sačuvano: eda_scatter_Is_Brzina.png")

# ── Grafik 3: Korelaciona matrica ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.drop(columns=['Klasa_naziv']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Korelaciona matrica karakteristika', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_korelacija.png', bbox_inches='tight')
plt.show()
print("Sačuvano: eda_korelacija.png")