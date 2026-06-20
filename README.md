# Detekcija kvara na asinhronom motoru

Projekat za predmet **Softverski algoritmi u sistemima automatskog upravljanja** (SAUSAU) — binarna klasifikacija režima rada asinhronog motora na osnovu simulacionih podataka iz MATLAB/Simulink.

## Opis problema

Cilj je detektovati gubitak jedne faze napajanja (open-phase fault) na asinhronom motoru, na osnovu fizičkih veličina dobijenih iz Simulink simulacije.

- **Klasa 0** — zdrav rad
- **Klasa 1** — gubitak faze (kvar)

## Podaci

Dataset (`matlab_simulacioni_podaci.csv`) sadrži 1400 uzoraka generisanih simulacijom naponskog matematičkog modela asinhrone mašine (`MModel_AM`), sa 8 ulaznih atributa:

`U_s`, `I_s`, `s`, `Vib`, `THD`, `P_f`, `Brzina`, `Temp`

## Struktura projekta

| Fajl | Opis |
|---|---|
| `priprema_podataka.py` | Učitavanje, čišćenje i standardizacija podataka |
| `eda.py` | Eksplorativna analiza — korelacije, boxplotovi, scatter |
| `logisticka_regresija.py` | Treniranje, hiperparametri, evaluacija |
| `knn.py` | Treniranje, hiperparametri, evaluacija |
| `feature_selection.py` | Odabir najznačajnijih atributa |
| `deployment.py` | Export modela + CLI za predikciju |

## Pokretanje

```bash
uv add pandas matplotlib seaborn scikit-learn joblib
uv run priprema_podataka.py
uv run eda.py
uv run logisticka_regresija.py
uv run knn.py
uv run feature_selection.py
uv run deployment.py
```

## Rezultati

Oba modela (logistička regresija i k-NN) postižu 100% tačnost na test skupu, što je posljedica jasne fizičke separacije između zdravog rada i kvara u simulacionim podacima.
