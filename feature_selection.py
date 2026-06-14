import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Ucitavanje pripremljenih podataka
X_train, X_test, y_train, y_test = joblib.load('pripremljeni_podaci.pkl')

# Ucitavanje naziva kolona
df = pd.read_csv('matlab_simulacioni_podaci.csv').drop(columns=['T', 'Klasa'])
nazivi_kolona = df.columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# POMOCNA FUNKCIJA — evaluacija modela
# ─────────────────────────────────────────────────────────────────────────────
def evaluiraj(naziv, y_test, y_pred):
    print(f"\n--- {naziv} ---")
    print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision :", round(precision_score(y_test, y_pred), 4))
    print("Recall    :", round(recall_score(y_test, y_pred), 4))
    print("F1-score  :", round(f1_score(y_test, y_pred), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — svi atributi
# ─────────────────────────────────────────────────────────────────────────────
print("=== BASELINE — svi atributi ===")

lr_baseline = LogisticRegression(max_iter=10000, solver='liblinear')
lr_baseline.fit(X_train, y_train)
evaluiraj("Logisticka regresija — svi atributi",
          y_test, lr_baseline.predict(X_test))

knn_baseline = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_baseline.fit(X_train, y_train)
evaluiraj("k-NN — svi atributi",
          y_test, knn_baseline.predict(X_test))

# ─────────────────────────────────────────────────────────────────────────────
# 2. METODA 1 — SelectKBest (statisticki test F)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SelectKBest (top 4 atributa) ===")

selector = SelectKBest(score_func=f_classif, k=4)
selector.fit(X_train, y_train)

X_train_kb = selector.transform(X_train)
X_test_kb  = selector.transform(X_test)

scores = pd.Series(selector.scores_, index=nazivi_kolona).sort_values(ascending=False)
print("\nF-score po atributima:")
print(scores.round(2))

odabrani_kb = [nazivi_kolona[i] for i in selector.get_support(indices=True)]
print(f"\nOdabrani atributi: {odabrani_kb}")

lr_kb = LogisticRegression(max_iter=10000, solver='liblinear')
lr_kb.fit(X_train_kb, y_train)
evaluiraj("Logisticka regresija — SelectKBest", y_test, lr_kb.predict(X_test_kb))

knn_kb = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_kb.fit(X_train_kb, y_train)
evaluiraj("k-NN — SelectKBest", y_test, knn_kb.predict(X_test_kb))

# ─────────────────────────────────────────────────────────────────────────────
# 3. METODA 2 — RFE (Recursive Feature Elimination)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== RFE — Recursive Feature Elimination (top 4 atributa) ===")

rfe = RFE(estimator=LogisticRegression(max_iter=10000, solver='liblinear'),
          n_features_to_select=4)
rfe.fit(X_train, y_train)

X_train_rfe = rfe.transform(X_train)
X_test_rfe  = rfe.transform(X_test)

rangiranje = pd.Series(rfe.ranking_, index=nazivi_kolona).sort_values()
print("\nRangiranje atributa (1 = odabran):")
print(rangiranje)

odabrani_rfe = [nazivi_kolona[i] for i in range(len(nazivi_kolona))
                if rfe.support_[i]]
print(f"\nOdabrani atributi: {odabrani_rfe}")

lr_rfe = LogisticRegression(max_iter=10000, solver='liblinear')
lr_rfe.fit(X_train_rfe, y_train)
evaluiraj("Logisticka regresija — RFE", y_test, lr_rfe.predict(X_test_rfe))

knn_rfe = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_rfe.fit(X_train_rfe, y_train)
evaluiraj("k-NN — RFE", y_test, knn_rfe.predict(X_test_rfe))

# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAFICI — poređenje
# ─────────────────────────────────────────────────────────────────────────────

# F-score atributa (SelectKBest)
plt.figure(figsize=(8, 5))
scores.plot(kind='barh', color='steelblue', edgecolor='black')
plt.xlabel('F-score')
plt.title('Vaznost atributa — SelectKBest (F-test)')
plt.tight_layout()
plt.savefig('fs_selectkbest.png')

# Poređenje F1 scoreova — svi vs odabrani atributi
metode = ['Svi atributi', 'SelectKBest', 'RFE']

f1_lr = [
    f1_score(y_test, lr_baseline.predict(X_test)),
    f1_score(y_test, lr_kb.predict(X_test_kb)),
    f1_score(y_test, lr_rfe.predict(X_test_rfe))
]

f1_knn = [
    f1_score(y_test, knn_baseline.predict(X_test)),
    f1_score(y_test, knn_kb.predict(X_test_kb)),
    f1_score(y_test, knn_rfe.predict(X_test_rfe))
]

x = np.arange(len(metode))
sirina = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - sirina/2, f1_lr,  sirina, label='Logisticka regresija',
        color='steelblue', edgecolor='black')
plt.bar(x + sirina/2, f1_knn, sirina, label='k-NN',
        color='tomato', edgecolor='black')
plt.xticks(x, metode)
plt.ylabel('F1-score')
plt.title('Poređenje F1-score: svi vs odabrani atributi')
plt.legend()
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('fs_poređenje.png')

print("\nFeature selection zavrsena. Sacuvane slike:")
print("  fs_selectkbest.png")
print("  fs_poređenje.png")