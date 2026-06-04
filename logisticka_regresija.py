import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
# 1. TRENIRANJE
# ─────────────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=10000, solver='liblinear')
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREDIKCIJA
# ─────────────────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────────────────────────────────────
# 3. METRIKE
# ─────────────────────────────────────────────────────────────────────────────
print("=== Logisticka regresija — Metrike ===")
print("Matrica konfuzije:")
print(confusion_matrix(y_test, y_pred))
print()
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
print("Precision :", round(precision_score(y_test, y_pred), 4))
print("Recall    :", round(recall_score(y_test, y_pred), 4))
print("F1-score  :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC   :", round(roc_auc_score(y_test, y_prob), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 4. KROS-VALIDACIJA
# ─────────────────────────────────────────────────────────────────────────────
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("\n=== Kros-validacija (5-fold) ===")
print("F1 po foldovima :", np.round(cv_scores, 4))
print("Srednja vrijednost:", round(cv_scores.mean(), 4))
print("Std devijacija    :", round(cv_scores.std(), 4))

# ─────────────────────────────────────────────────────────────────────────────
# 5. PODEŠAVANJE HIPERPARAMETARA — parametar C
#    C kontrolise regularizaciju:
#    - mali C → jaca regularizacija, jednostavniji model
#    - veliki C → slabija regularizacija, model se vise prilagođava podacima
# ─────────────────────────────────────────────────────────────────────────────
C_vrijednosti = [0.001, 0.01, 0.1, 1, 10, 100]
f1_scores = []

for C in C_vrijednosti:
    m = LogisticRegression(C=C, max_iter=10000, solver='liblinear')
    score = cross_val_score(m, X_train, y_train, cv=5, scoring='f1').mean()
    f1_scores.append(score)
    print(f"C = {C:>6} | F1 = {score:.4f}")

najbolji_C = C_vrijednosti[np.argmax(f1_scores)]
print(f"\nNajbolji C: {najbolji_C}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FINALNI MODEL sa najboljim C
# ─────────────────────────────────────────────────────────────────────────────
finalni_model = LogisticRegression(C=najbolji_C, max_iter=10000, solver='liblinear')
finalni_model.fit(X_train, y_train)
y_pred_final = finalni_model.predict(X_test)
y_prob_final = finalni_model.predict_proba(X_test)[:, 1]

print("\n=== Finalni model (C={}) ===".format(najbolji_C))
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
disp.plot(cmap='Blues', colorbar=False)
plt.title('Matrica konfuzije — Logisticka regresija')
plt.tight_layout()
plt.savefig('lr_matrica_konfuzije.png')
plt.show()

# ROC kriva
fpr, tpr, _ = roc_curve(y_test, y_prob_final)
auc = roc_auc_score(y_test, y_prob_final)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'Logisticka regresija (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Slucajni klasifikator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC kriva — Logisticka regresija')
plt.legend()
plt.tight_layout()
plt.savefig('lr_roc_kriva.png')
plt.show()

# F1 u zavisnosti od C
plt.figure(figsize=(7, 5))
plt.plot(C_vrijednosti, f1_scores, marker='o', color='steelblue', linewidth=2)
plt.xscale('log')
plt.xlabel('Vrijednost C (log skala)')
plt.ylabel('F1-score (5-fold CV)')
plt.title('Podesavanje hiperparametra C — Logisticka regresija')
plt.tight_layout()
plt.savefig('lr_hiperparametri.png')
plt.show()

# Cuvanje modela
joblib.dump(finalni_model, 'model_logisticka_regresija.pkl')
print("\nModel sacuvan: model_logisticka_regresija.pkl")