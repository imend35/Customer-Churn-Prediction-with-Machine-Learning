# =========================
# 1. Kütüphaneler
# =========================
import pandas as pd  # Veri tabloları üzerinde işlem yapmak için (Excel gibi düşünün)
import numpy as np   # Sayısal hesaplamalar ve matris işlemleri için

from sklearn.model_selection import train_test_split # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.preprocessing import LabelEncoder       # Kategorik verileri (Kız/Erkek vb.) sayılara (0/1) çevirmek için
from sklearn.ensemble import RandomForestClassifier  # Random Forest sınıflandırma modelini kullanmak için
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Modelin başarısını ölçmek için

# =========================
# 2. Veri Yükleme
# =========================
df = pd.read_csv("telco_churn.csv") # Belirtilen yoldaki veriyi okur ve tabloya aktarır

print("Veri boyutu:", df.shape) # Verinin kaç satır ve sütundan oluştuğunu kontrol eder
print(df.head())              # Verinin genel yapısını görmek için ilk 5 satırı basar

# =========================
# 3. Veri Temizleme
# =========================

# 'customerID' tahminde bir anlam ifade etmediği için (herkese özel numara) veri setinden atılır
df.drop("customerID", axis=1, inplace=True)

# 'TotalCharges' metin olarak gelmiş olabilir, bunu sayıya çevirir. Sayıya dönmeyen hatalı kısımları boşluk (NaN) yapar.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Dönüşümden sonra veya öncesinde oluşan boş (eksik) verilerin bulunduğu satırları siler
df.dropna(inplace=True)

# Sadece iki kategorisi olan (Binary) sütunların listesi
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

le = LabelEncoder() # Yazıları sayıya çevirecek araç
for col in binary_cols:
    if col in df.columns:
        # Örn: 'Churn' sütunundaki "Yes/No" yazılarını "1/0" sayılarına dönüştürür
        df[col] = le.fit_transform(df[col])

# İkiden fazla kategoriye sahip sütunları (Örn: Ödeme Türü) her bir kategori için ayrı sütuna (0/1) böler
df = pd.get_dummies(df, drop_first=True)

# =========================
# 4. Feature & Target (Özellikler ve Hedef)
# =========================
X = df.drop("Churn", axis=1) # Bağımsız değişkenler (Tahminde kullanılacak girdiler)
y = df["Churn"]              # Bağımlı değişken (Tahmin edilmeye çalışılan sonuç)

# Veriyi %80 eğitim, %20 test olacak şekilde ayırır.
# 'stratify=y' sayesinde eğitim ve testteki 'Churn' olma oranı birbirine eşit tutulur.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape) # Eğitilecek veri miktarını gösterir
print("Test shape:", X_test.shape)   # Test edilecek veri miktarını gösterir

# =========================
# 5. Random Forest Model
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,    # Toplamda 200 adet bağımsız karar ağacı oluşturulur
    max_depth=10,        # Ağaçların çok derinleşip veriyi ezberlemesini (overfitting) önlemek için sınır koyar
    random_state=42,     # Rastgelelik içeren işlemlerde her seferinde aynı sonucun alınmasını sağlar
    class_weight="balanced" # Dengesiz verilerde (örn: Churn sayısı çok azsa) sınıflara adil davranılmasını sağlar
)

# Modeli eğitim verileriyle besleyerek öğrenme sürecini başlatır
rf_model.fit(X_train, y_train)

# =========================
# 6. Tahmin
# =========================
y_pred_rf = rf_model.predict(X_test)         # Modelin test verilerine dayanarak yaptığı net sınıf tahminleri (0 veya 1)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1] # Modelin müşterinin gideceğine dair hesapladığı olasılık değeri (0.0 - 1.0 arası)

print("\nİlk 10 tahmin:", y_pred_rf[:10])   # İlk 10 müşterinin tahmin edilen sonuçlarını gösterir
print("İlk 10 olasılık:", y_prob_rf[:10]) # İlk 10 müşterinin kaç olasılıkla gideceğini gösterir

# =========================
# 7. Performans
# =========================
print("\n=== RANDOM FOREST PERFORMANSI ===")

# Toplam doğru tahmin sayısının toplam veriye oranı
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Modelin detaylı karnesi (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Kaç doğru, kaç yanlış tahmin yapıldığını gösteren matris (Hata tablosu)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# =========================
# 8. Feature Importance (Özellik Önem Düzeyi)
# =========================
# Hangi değişkenin (Örn: İnternet paketi, aylık ödeme) tahminde ne kadar payı olduğunu hesaplar
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

# Önem düzeyine göre büyükten küçüğe sıralar
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10)) # Sonucu en çok etkileyen 10 değişkeni yazdırır

# =========================
# 9. Threshold Ayarı (Bonus)
# =========================
# Normalde 0.5 olan sınırı 0.3'e indirir.
# Amaç: Müşterinin gitme olasılığı %30 bile olsa onu "gidebilir" olarak işaretleyip erkenden önlem almaktır.
threshold = 0.3
y_pred_custom = (y_prob_rf > threshold).astype(int)

print("\nCustom Threshold Confusion Matrix (0.3):")
print(confusion_matrix(y_test, y_pred_custom)) # Yeni eşik değerine göre modelin performansını ölçer