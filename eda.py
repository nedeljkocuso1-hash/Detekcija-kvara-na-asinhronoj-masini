import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
import joblib

# Ucitavanje podataka
df = pd.read_csv('matlab_simulacioni_podaci.csv')
df = df.drop(columns=['T'])

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROVJERA ANOMALIJA I NEDOSTAJUCIH VRIJEDNOSTI
# ─────────────────────────────────────────────────────────────────────────────
print("=== Nedostajuce vrijednosti ===")
print(df.isnull().sum())

print("\n=== Osnovna statistika ===")
print(df.describe())

# ─────────────────────────────────────────────────────────────────────────────
# 2. KORELACIONA MATRICA
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
korelacija = df.corr()
s.heatmap(korelacija, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5)
plt.title("Korelaciona matrica karakteristika")
plt.tight_layout()
plt.savefig('eda_korelacija.png')

# Ispis jakih korelacija (>0.8 ili <-0.8), iskljucujuci dijagonalu
print("\n=== Jake korelacije (|r| > 0.8) ===")
for col in korelacija.columns:
    for idx in korelacija.index:
        if col != idx:
            val = korelacija.loc[idx, col]
            if abs(val) > 0.8:
                print(f"{idx} <-> {col} : {val:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BOXPLOTOVI — uocavanje outliera i razlike izmedju klasa
# ─────────────────────────────────────────────────────────────────────────────
features = [c for c in df.columns if c != 'Klasa']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feat in enumerate(features):
    df.boxplot(column=feat, by='Klasa', ax=axes[i])
    axes[i].set_title(feat)
    axes[i].set_xlabel('Klasa (0 = Zdrav, 1 = Kvar)')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplot karakteristika po klasama')
plt.tight_layout()
plt.savefig('eda_boxplots.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. DISTRIBUCIJA KLASA
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 4))
df['Klasa'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'],
                                 edgecolor='black')
plt.title('Distribucija klasa')
plt.xlabel('Klasa (0 = Zdrav, 1 = Kvar)')
plt.ylabel('Broj uzoraka')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda_distribucija_klasa.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. SCATTER PLOT — I_s vs Brzina (najjaca separacija)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for klasa, boja, naziv in [(0, 'steelblue', 'Zdrav rad'), 
                            (1, 'tomato', 'Gubitak faze')]:
    podskup = df[df['Klasa'] == klasa]
    plt.scatter(podskup['I_s'], podskup['Brzina'],
                label=naziv, alpha=0.7, color=boja)

plt.xlabel('Struja statora I_s [A]')
plt.ylabel('Brzina rotora [rpm]')
plt.title('Separacija klasa: I_s vs Brzina')
plt.legend()
plt.tight_layout()
plt.savefig('eda_scatter.png')

print("\nEDA zavrsena. Sacuvane slike:")
print("  eda_korelacija.png")
print("  eda_boxplots.png")
print("  eda_distribucija_klasa.png")
print("  eda_scatter.png")