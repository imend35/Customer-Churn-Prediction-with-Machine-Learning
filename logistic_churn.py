# =========================
# 1. Kütüphaneler
# =========================
import pandas as pd  # Tablosal veri işlemleri ve veri çerçeveleri (DataFrames) için
import numpy as np   # Sayısal hesaplamalar ve dizi (array) işlemleri için

from sklearn.model_selection import train_test_split # Veriyi eğitim ve test olarak bölmek için
from sklearn.preprocessing import LabelEncoder       # Kategorik metinleri sayısal değerlere dönüştürmek için
from sklearn.linear_model import LogisticRegression   # Sınıflandırma algoritmasını (Lojistik Regresyon) kullanmak için
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Başarı metriklerini hesaplamak için

# =========================
# 2. Veri Yükleme
# =========================
df = pd.read_csv("telco_churn.csv") # CSV formatındaki veri setini okuyup bir DataFrame'e aktarır

print("Veri boyutu:", df.shape) # Verinin kaç satır ve kaç sütundan oluştuğunu yazdırır
print(df.head())              # Verinin ilk 5 satırını gözlemlemek için ekrana basar

# =========================
# 3. Veri Temizleme
# =========================

# Tahminde etkisi olmayacak benzersiz müşteri kimlik numarasını veri setinden çıkarır
df.drop("customerID", axis=1, inplace=True)

# 'TotalCharges' sütununu sayısal formata çevirir; sayıya çevrilemeyen hatalı değerleri (boşluk gibi) NaN yapar
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Dönüşüm sonrası oluşan veya mevcut olan eksik verilerin (NaN) olduğu satırları tamamen siler
df.dropna(inplace=True)

# Sadece iki farklı değer içeren (Evet/Hayır gibi) kategorik sütunların listesi
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

le = LabelEncoder() # Dönüştürücü nesneyi oluşturur
for col in binary_cols:
    if col in df.columns:
        # Metin değerlerini (örn: Male/Female) 0 ve 1 gibi sayılara dönüştürür
        df[col] = le.fit_transform(df[col])

# İkiden fazla kategori içeren sütunları (örn: Ödeme yöntemi) kukla (dummy) değişkenlere ayırır (0-1 matrisi)
df = pd.get_dummies(df, drop_first=True)

# =========================
# 4. Feature & Target
# =========================
X = df.drop("Churn", axis=1) # Hedef değişken dışındaki tüm sütunları (bağımsız değişkenler) X'e atar
y = df["Churn"]              # Tahmin edilmek istenen 'Churn' sütununu (hedef değişken) y'ye atar

# Veriyi %80 eğitim, %20 test olarak böler. 
# 'stratify=y' parametresi, hedef değişkenin oranını (churn olan/olmayan) her iki grupta da korur.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape) # Eğitim setinin boyutunu yazdırır
print("Test shape:", X_test.shape)   # Test setinin boyutunu yazdırır

# =========================
# 5. Model Kurulumu
# =========================
# Lojistik Regresyon modelini tanımlar; 'max_iter' modelin çözüme ulaşması için yapacağı maksimum deneme sayısıdır
model = LogisticRegression(max_iter=1000)

# Modeli eğitim verilerini (X_train) ve doğru cevapları (y_train) kullanarak eğitir
model.fit(X_train, y_train)

# =========================
# 6. Tahmin
# =========================
y_pred = model.predict(X_test)         # Test verilerine bakarak 0 veya 1 şeklinde sınıf tahmini yapar
y_prob = model.predict_proba(X_test)[:, 1] # Tahminin 1 (Churn) olma olasılığını (0 ile 1 arası değer) hesaplar

print("\nİlk 10 tahmin:", y_pred[:10])   # İlk 10 örnek için yapılan sınıf tahminlerini gösterir
print("İlk 10 olasılık:", y_prob[:10]) # İlk 10 örnek için hesaplanan olasılık değerlerini gösterir

# =========================
# 7. Performans
# =========================
print("\n=== MODEL PERFORMANSI ===")

# Modelin genel doğru tahmin oranını hesaplar
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision (kesinlik), Recall (duyarlılık) ve F1 skorunu detaylı rapor halinde sunar
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Gerçek değerler ile tahmin edilenlerin karşılaştırmalı tablosunu (Hata Matrisi) oluşturur
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 8. Feature Importance (Katsayı Analizi)
# =========================
# Modelin her bir değişkene verdiği katsayıları (ağırlıkları) içeren bir tablo oluşturur
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

# Katsayıları büyükten küçüğe sıralayarak hangi faktörlerin sonucu daha çok etkilediğini belirler
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

print("\nTop 10 Positive (Churn artıran):") # Katsayısı pozitif olanlar churn ihtimalini artırır
print(coefficients.head(10))

print("\nTop 10 Negative (Churn azaltan):") # Katsayısı negatif olanlar müşterinin kalma ihtimalini artırır
print(coefficients.tail(10))

# =========================
# 9. Threshold Ayarı (Bonus)
# =========================
# Varsayılan 0.5 olan eşik değerini 0.3'e çeker.
# Bu, "en ufak bir risk gördüğünde bile churn diye işaretle" demektir (Daha hassas model).
threshold = 0.3
y_pred_custom = (y_prob > threshold).astype(int)

print("\nCustom Threshold Confusion Matrix (0.3):")
print(confusion_matrix(y_test, y_pred_custom)) # Yeni eşik değerine göre hata matrisini yazdırır