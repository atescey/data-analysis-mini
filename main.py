import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("data/training.csv")

print("İlk 5 satır:")
print(df.head())

print("\nVeri setinin boyutu:")
print(df.shape)

print("\nEksik değer sayıları:")
print(df.isnull().sum())

print("\nİstatistiksel özet:")
print(df.describe())

print("\nEtiket dağılımı:")
print(df['Label'].value_counts())

plt.figure(figsize=(6,4))
plt.hist(df['Label'], bins=2)
plt.title("Etiket Dağılımı (0 = Background, 1 = Signal)")
plt.xlabel("Etiket")
plt.ylabel("Frekans")

plot_path = "plots/plot1.png"
plt.savefig(plot_path)
plt.close()

print(f"\nGrafik kaydedildi: {plot_path}")

if "Weight" in df.columns:
    df = df.drop(columns=["Weight"])

y = df["Label"]
X = df.drop(columns=["Label"])

X = X.replace(-999.0, np.nan)
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTrain/Test hazırlığı tamam.")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("\nModel eğitiliyor...")
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Sonuçlar
accuracy = accuracy_score(y_test, y_pred)
print("\nDoğruluk:", accuracy)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_curve, auc

# ================================
# 1) ROC Curve oluşturma
# ================================

# RandomForest probability çıktıları
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

roc_path = "plots/roc_curve.png"
plt.savefig(roc_path)
plt.close()

print(f"ROC grafiği kaydedildi: {roc_path}")

# ================================
# 2) Feature Importance grafiği
# ================================

importances = model.feature_importances_
indices = np.argsort(importances)[-15:]  # en önemli 15 özellik

plt.figure(figsize=(6, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), X.columns[indices])
plt.xlabel("Önem Skoru")
plt.title("Feature Importance (Top 15)")

fi_path = "plots/feature_importance.png"
plt.savefig(fi_path)
plt.close()

print(f"Feature Importance grafiği kaydedildi: {fi_path}")
