import yfinance as yf
import pandas as pd

# 1. Configuration de l'Univers final stabilisé
anchor = ["HG=F"]
proxies = ["CPER", "JJCTF", "PICK", "XME"]
equities = [
    "FCX", "SCCO", "IVPAF", "LUNMF", "TECK", 
    "HBM", "ERO", "AAUKF", "BHP", "RIO",
    "GLNCY", "ANFGF", "FM.TO", "ZIJMF", "TGB"
]

full_universe = anchor + proxies + equities

# 2. Paramètres de dates
start_date = "2020-01-01"
end_date = "2026-01-01"

print(f"Téléchargement de {len(full_universe)} actifs du {start_date} au {end_date}...")

# 3. Téléchargement des données
raw_data = yf.download(
    full_universe, 
    start=start_date, 
    end=end_date, 
    auto_adjust=True
)

# 4. Extraction des prix (Close)
prices = raw_data['Close']

# 5. Nettoyage des données
# On remplit les trous (jours fériés différents) par la méthode "Forward Fill"
# Puis on supprime les lignes totalement vides (week-ends)
prices = prices.ffill().dropna(how='all')

# 6. Exportation en CSV
file_name = "copper_universe_data_2020_2026.csv"
prices.to_csv(file_name)

print("-" * 30)
print(f"Succès ! Fichier enregistré sous : {file_name}")
print(f"Nombre de lignes : {len(prices)}")
print(f"Nombre de colonnes : {len(prices.columns)}")
print("-" * 30)

# Petit aperçu pour vérifier
print(prices.head())