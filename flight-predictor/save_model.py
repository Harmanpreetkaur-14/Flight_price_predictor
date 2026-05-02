# save_model.py  — run this ONCE after training, inside your notebook environment
# Place this file next to your notebook, then run:  python save_model.py

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ── 1. Load & preprocess (mirrors your notebook exactly) ──────────────────────
df = pd.read_csv("Clean_Dataset.csv")
df = df.drop(['Unnamed: 0', 'flight'], axis=1)
df['class'] = df['class'].map({'Business': 1, 'Economy': 0})
df['stops'] = pd.factorize(df['stops'])[0]

df = df.join(pd.get_dummies(df.airline,        prefix='airline',          dtype=int)).drop('airline', axis=1)
df = df.join(pd.get_dummies(df.source_city,    prefix='source_city',      dtype=int)).drop('source_city', axis=1)
df = df.join(pd.get_dummies(df.destination_city,prefix='destination_city',dtype=int)).drop('destination_city', axis=1)
df = df.join(pd.get_dummies(df.arrival_time,   prefix='arrival_time',     dtype=int)).drop('arrival_time', axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure_time',   dtype=int)).drop('departure_time', axis=1)

X = df.drop('price', axis=1)
y = df['price']

# ── 2. Save the exact column order (critical for prediction) ──────────────────
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("Columns saved:", list(X.columns))

# ── 3. Train & save model ─────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
reg.fit(x_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(reg, f)

print("✅ model.pkl saved successfully!")
print(f"   Training R² : {reg.score(x_train, y_train):.4f}")
print(f"   Test     R² : {reg.score(x_test,  y_test):.4f}")

# ── 4. Copy both files to your Flask project ─────────────────────────────────
# cp model.pkl flight-predictor/
# cp model_columns.pkl flight-predictor/
