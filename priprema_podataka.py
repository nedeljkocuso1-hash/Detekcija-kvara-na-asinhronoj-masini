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
# ─────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=['Klasa'])
y = df['Klasa']

print("=== Karakteristike i ciljna varijabla ===")
print(f"X oblik    : {X.shape}  → {list(X.columns)}")
print(f"y oblik    : {y.shape}")
print(f"Distribucija klasa:\n{y.value_counts().rename({0: 'Zdrav (0)', 1: 'Kvar (1)'})}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Podjela na trening i test set
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
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)   
X_test_sc  = scaler.transform(X_test)        

# Vrati u DataFrame radi preglednosti (čuvaju se nazivi kolona)
X_train_sc = pd.DataFrame(X_train_sc, columns=X.columns)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=X.columns)

print("=== Skaliranje — provjera (srednja vrijednost i std na treningu) ===")
print("Trening skup nakon skaliranja:")
print(X_train_sc.mean().round(6).to_string())  
print()
print(X_train_sc.std().round(6).to_string())    
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Čuvanje pripremljenih podataka
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(
    (X_train_sc, X_test_sc, y_train, y_test),
    'pripremljeni_podaci.pkl'
)
joblib.dump(scaler, 'scaler.pkl')