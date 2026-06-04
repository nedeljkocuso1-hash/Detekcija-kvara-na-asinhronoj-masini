"""
priprema_podataka.py
====================
Priprema podataka za binarnu klasifikaciju kvara na asinhronom motoru.
Algoritmi koji slijede: Logistička regresija i k-NN.

Teorijska osnova:
- Nadgledano učenje, binarna klasifikacija: y ∈ {0, 1}
- StandardScaler (Z-score): x' = (x - μ) / σ
- Fit SAMO na trening setu → izbjegavamo data leakage
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. Učitavanje dataseta
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('matlab_simulacioni_podaci.csv')

print("=== Originalni dataset ===")
print(f"Dimenzije       : {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone          : {list(df.columns)}")
print(f"NaN vrijednosti :\n{df.isnull().sum()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Čišćenje — uklanjamo kolonu T (potpuno NaN, Simulink greška u vezama)
# ─────────────────────────────────────────────────────────────────────────────
df = df.drop(columns=['T'])

print("=== Nakon čišćenja ===")
print(f"Dimenzije : {df.shape[0]} redova x {df.shape[1]} kolona")
print(f"Kolone    : {list(df.columns)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Odvajanje karakteristika (X) od ciljne varijable (y)
#    X → fizičke veličine motora (features)
#    y → diskretna klasa: 0 = zdrav rad, 1 = gubitak faze
# ─────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=['Klasa'])
y = df['Klasa']

print("=== Karakteristike i ciljna varijabla ===")
print(f"X oblik    : {X.shape}  → {list(X.columns)}")
print(f"y oblik    : {y.shape}")
print(f"Distribucija klasa:\n{y.value_counts().rename({0: 'Zdrav (0)', 1: 'Kvar (1)'})}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Podjela na trening i test set
#
#    Koristimo odnos 80:20 jer:
#    - Dataset ima 200 uzoraka — relativno mali skup
#    - 80% treninga (160 uzoraka) daje modelu dovoljno primjera za učenje
#    - 20% testa  ( 40 uzoraka) daje pouzdanu procjenu generalizacije
#    - Za veće skupove (>1000) mogao bi se koristiti 70:30
#
#    stratify=y → osigurava isti omjer klasa u oba skupa (po 50/50)
#    random_state=42 → reproduktivnost rezultata
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=== Podjela na trening/test set (80:20) ===")
print(f"Trening skup : {X_train.shape[0]} uzoraka")
print(f"Test skup    : {X_test.shape[0]} uzoraka")
print(f"Klase u treningu : {y_train.value_counts().to_dict()}")
print(f"Klase u testu    : {y_test.value_counts().to_dict()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Skaliranje — StandardScaler (Z-score normalizacija)
#
#    Zašto StandardScaler a ne MinMax?
#    - k-NN koristi Euklidsko rastojanje → dominiraju karakteristike
#      velikog opsega (U_s ≈ 400, Brzina ≈ 1500) nad malim (Vib ≈ 0.01)
#    - StandardScaler svodi sve na μ=0, σ=1 → ravnopravne karakteristike
#    - Logistička regresija konvergira brže sa standardizovanim podacima
#    - MinMax je osjetljiviji na outlier-e; StandardScaler je robusniji
#
#    VAŽNO — Data leakage pravilo:
#    - scaler.fit_transform(X_train) → uči μ i σ SAMO iz trening seta
#    - scaler.transform(X_test)      → primjenjuje iste μ i σ na test set
#    - NE smijemo fitovati na X_test jer bi model "vidio" test podatke
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)   # uči + transformiše
X_test_sc  = scaler.transform(X_test)        # samo transformiše

# Vrati u DataFrame radi preglednosti (čuvaju se nazivi kolona)
X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=X.columns)

print("=== Skaliranje — provjera (srednja vrijednost i std na treningu) ===")
print("Trening skup nakon skaliranja:")
print(X_train_sc.mean().round(6).to_string())   # sve ≈ 0.0
print()
print(X_train_sc.std().round(6).to_string())    # sve ≈ 1.0
print()
print("Napomena: Na test setu μ i σ neće biti tačno 0/1 — to je normalno.")
print("          Test set skaliramo parametrima naučenim iz trening seta.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Čuvanje pripremljenih podataka
#    Sačuvane promenljive učitavamo u sledećim skriptama (02, 03)
#    bez ponovnog ponavljanja cijelog procesa pripreme
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(
    (X_train_sc, X_test_sc, y_train, y_test),
    'pripremljeni_podaci.pkl'
)
joblib.dump(scaler, 'scaler.pkl')

print("=== Sačuvano ===")
print("pripremljeni_podaci.pkl  → skalirani X_train, X_test, y_train, y_test")
print("scaler.pkl               → obučeni StandardScaler")