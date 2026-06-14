import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# Ucitavanje pripremljenih podataka
X_train, X_test, y_train, y_test = joblib.load('pripremljeni_podaci.pkl')

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRENIRANJE (pocetni k=5)
# ─────────────────────────────────────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDIKCIJA
# ─────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_train = model.predict(X_train)

# ─────────────────────────────────────────────────────────────────────────────
# 3. METRIKE
# ─────────────────────────────────────────────────────────────────────────────
print("=== k-NN (k=5) — Metrike ===")
print("Matrica konfuzije:")
print(confusion_matrix(y_test, y_pred))
print()
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
print("Precision :", round(precision_score(y_test, y_pred), 4))
print("Recall    :", round(recall_score(y_test, y_pred), 4))
print("F1-score  :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC   :", round(roc_auc_score(y_test, y_prob), 4))
print()
print("Tačnost na TRENING skupu:", round(accuracy_score(y_train, y_pred_train), 4))
print("Tačnost na TEST skupu:", round(accuracy_score(y_test, y_pred), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 4. KROS-VALIDACIJA
# ─────────────────────────────────────────────────────────────────────────────
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("\n=== Kros-validacija (5-fold) ===")
print("F1 po foldovima :", np.round(cv_scores, 4))
print("Srednja vrijednost:", round(cv_scores.mean(), 4))
print("Std devijacija    :", round(cv_scores.std(), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 5. PODESAVANJE HIPERPARAMETARA — trazenje najboljeg k
# ─────────────────────────────────────────────────────────────────────────────
k_vrijednosti = range(1, 21)
f1_scores = []

for k in k_vrijednosti:
    m = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    score = cross_val_score(m, X_train, y_train, cv=5, scoring='f1').mean()
    f1_scores.append(score)
    print(f"k = {k:>2} | F1 = {score:.4f}")

najbolji_k = k_vrijednosti[np.argmax(f1_scores)]
print(f"\nNajbolji k: {najbolji_k}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINALNI MODEL sa najboljim k
# ─────────────────────────────────────────────────────────────────────────────
finalni_model = KNeighborsClassifier(n_neighbors=najbolji_k, metric='euclidean')
finalni_model.fit(X_train, y_train)
y_pred_final = finalni_model.predict(X_test)
y_prob_final = finalni_model.predict_proba(X_test)[:, 1]

print("\n=== Finalni model (k={}) ===".format(najbolji_k))
print("Accuracy  :", round(accuracy_score(y_test, y_pred_final), 4))
print("Precision :", round(precision_score(y_test, y_pred_final), 4))
print("Recall    :", round(recall_score(y_test, y_pred_final), 4))
print("F1-score  :", round(f1_score(y_test, y_pred_final), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAFICI
# ─────────────────────────────────────────────────────────────────────────────

# Matrica konfuzije
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(cm, display_labels=['Zdrav (0)', 'Kvar (1)'])
disp.plot(cmap='Oranges', colorbar=False)
plt.title('Matrica konfuzije — k-NN (k={})'.format(najbolji_k))
plt.tight_layout()
plt.savefig('knn_matrica_konfuzije.png')

# ROC kriva
fpr, tpr, _ = roc_curve(y_test, y_prob_final)
auc = roc_auc_score(y_test, y_prob_final)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='tomato', linewidth=2,
         label=f'k-NN (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Slucajni klasifikator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC kriva — k-NN')
plt.legend()
plt.tight_layout()
plt.savefig('knn_roc_kriva.png')

# F1 u zavisnosti od k
plt.figure(figsize=(7, 5))
plt.plot(k_vrijednosti, f1_scores, marker='o', color='tomato', linewidth=2)
plt.xlabel('Vrijednost k')
plt.ylabel('F1-score (5-fold CV)')
plt.title('Podesavanje hiperparametra k — k-NN')
plt.xticks(k_vrijednosti)
plt.tight_layout()
plt.savefig('knn_hiperparametri.png')

# Cuvanje modela
joblib.dump(finalni_model, 'model_knn.pkl')
print("\nModel sacuvan: model_knn.pkl")