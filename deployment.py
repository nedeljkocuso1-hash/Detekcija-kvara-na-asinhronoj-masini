import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#  OVO GASI UPOZORENJA
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPORT MODELA
#    Modeli su vec sacuvani u prethodnim skriptama, ali ovdje
#    ucitavamo oba i cuvamo zajedno sa scalerom u jedan paket
# ─────────────────────────────────────────────────────────────────────────────
scaler = joblib.load('scaler.pkl')
model_lr  = joblib.load('model_logisticka_regresija.pkl')
model_knn = joblib.load('model_knn.pkl')

paket = {
    'scaler'     : scaler,
    'logisticka' : model_lr,
    'knn'        : model_knn
}

joblib.dump(paket, 'motor_fault_detection.pkl')
print("Modeli exportovani: motor_fault_detection.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FUNKCIJA ZA PREDIKCIJU
#    Ovo je srce deployment-a — prima sirove vrijednosti senzora,
#    skalira ih i vraca predikciju sa vjerovatnocom
# ─────────────────────────────────────────────────────────────────────────────
def predikuj(U_s, I_s, s, Vib, THD, P_f, Brzina, Temp, model='logisticka'):
    """
    Parametri:
        U_s    — Efektivni napon statora [V]
        I_s    — RMS struja statora [A]
        s      — Klizanje
        Vib    — Vibracije
        THD    — Total Harmonic Distortion
        P_f    — Faktor snage
        Brzina — Mehanicka brzina rotora [rpm]
        Temp   — Temperatura [°C]
        model  — 'logisticka' ili 'knn'

    Vraca:
        klasa      — 0 (zdrav) ili 1 (kvar)
        vjerovatnoca — vjerovatnoca kvara [0.0 - 1.0]
    """
    paket = joblib.load('motor_fault_detection.pkl')
    scaler = paket['scaler']
    m      = paket[model]

    #ulaz = np.array([[U_s, I_s, s, Vib, THD, P_f, Brzina, Temp]])
    ulaz = pd.DataFrame([[U_s, I_s, s, Vib, THD, P_f, Brzina, Temp]], columns=['U_s', 'I_s', 's', 'Vib', 'THD', 'P_f', 'Brzina', 'Temp'])
    ulaz_skaliran = scaler.transform(ulaz)

    klasa        = m.predict(ulaz_skaliran)[0]
    vjerovatnoca = m.predict_proba(ulaz_skaliran)[0][1]

    return klasa, round(vjerovatnoca, 4)

# ─────────────────────────────────────────────────────────────────────────────
# 3. JEDNOSTAVAN UI — komandna linija
# ─────────────────────────────────────────────────────────────────────────────
def ui():
    print("\n" + "="*50)
    print("   DETEKCIJA KVARA NA ASINHRONOM MOTORU")
    print("="*50)

    print("\nOdaberi model:")
    print("  1 — Logisticka regresija")
    print("  2 — k-NN")
    izbor = input("Unos (1/2): ").strip()
    model = 'logisticka' if izbor == '1' else 'knn'

    print("\nUnesi vrijednosti senzora:")
    try:
        U_s    = float(input("  Napon statora U_s [V]         : "))
        I_s    = float(input("  Struja statora I_s [A]        : "))
        s      = float(input("  Klizanje s                    : "))
        Vib    = float(input("  Vibracije Vib                 : "))
        THD    = float(input("  THD                           : "))
        P_f    = float(input("  Faktor snage P_f              : "))
        Brzina = float(input("  Brzina rotora [rpm]           : "))
        Temp   = float(input("  Temperatura Temp              : "))
    except ValueError:
        print("\nGreska: unesi brojeve!")
        return

    klasa, vjerovatnoca = predikuj(
        U_s, I_s, s, Vib, THD, P_f, Brzina, Temp, model=model
    )

    print("\n" + "="*50)
    print("   REZULTAT PREDIKCIJE")
    print("="*50)

    if klasa == 0:
        print("  Status      : ZDRAV RAD")
    else:
        print("  Status      : ⚠ KVAR DETEKTOVAN — Gubitak faze!")

    print(f"  Vjerovatnoca kvara : {vjerovatnoca * 100:.2f}%")
    print(f"  Model              : {model}")
    print("="*50)

    # Opcija za ponovni unos
    ponovo = input("\nNova predikcija? (da/ne): ").strip().lower()
    if ponovo == 'da':
        ui()

# ─────────────────────────────────────────────────────────────────────────────
# 4. POKRETANJE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ui()