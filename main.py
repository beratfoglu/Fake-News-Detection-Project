import pandas as pd
import numpy as np
import re  # Metin temizliği için
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print("---------------------------------------------------------")
print("PROJE BAŞLATILIYOR... (Geliştirilmiş Versiyon)")
print("---------------------------------------------------------")

# --- 1. VERİLERİ YÜKLEME ---
print(">> Dosyalar okunuyor...")
try:
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("HATA: 'True.csv' veya 'Fake.csv' bulunamadı!")
    exit()

# Etiketleme
df_true['label'] = 0  # Gerçek
df_fake['label'] = 1  # Sahte

# --- ÖNEMLİ DÜZELTME 1: "Reuters" Hilesini Temizle ---
# Modelin sadece "Reuters" kelimesine bakıp karar vermesini engelliyoruz.
print(">> Veri temizliği yapılıyor (Reuters etiketleri kaldırılıyor)...")
try:
    # Sadece True.csv içinde genelde bu format olur
    df_true['text'] = df_true['text'].apply(lambda x: re.sub(r"^.*?-\s", "", x) if isinstance(x, str) else x)
except:
    pass # Hata verirse geç (farklı format olabilir)

# --- ÖNEMLİ DÜZELTME 2: Başlık ve Metni Birleştir ---
# Modelin hem başlığı hem de metni öğrenmesini sağlıyoruz.
# Böylece sen sadece başlık girdiğinde de doğru tahmin yapabilir.
df_true['combined_text'] = df_true['title'] + " " + df_true['text']
df_fake['combined_text'] = df_fake['title'] + " " + df_fake['text']

# --- 2. VERİLERİ BİRLEŞTİRME ---
dataframe = pd.concat([df_true, df_fake])
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

print(f">> Toplam Haber Sayısı: {len(dataframe)}")

# --- 3. HAZIRLIK ---
# Artık sadece 'text' değil, 'combined_text' kullanıyoruz
x = dataframe['combined_text'] 
y = dataframe['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# --- 4. VEKTÖRLEŞTİRME ---
print(">> Metinler vektörleştiriliyor...")
# max_df=0.7 -> Kelime belgelerin %70'inden fazlasında geçiyorsa (çok yaygınsa) at
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# --- 5. MODEL EĞİTİMİ ---
print(">> Model eğitiliyor...")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# --- 6. BAŞARI TESTİ ---
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
basari_yuzdesi = round(score * 100, 2)

print("\n" + "="*40)
print(f"✅ MODEL GÜNCELLENDİ!")
print(f"📊 BAŞARI ORANI: %{basari_yuzdesi}")
print("="*40)

# --- 7. TEST ---
print("\nNOT: Bu veri seti İNGİLİZCE haberler üzerine kuruludur.")
print("Lütfen test ederken İNGİLİZCE başlık veya metin giriniz.")
print("-" * 40)

while True:
    print("\n[Çıkmak için 'q' yazıp Enter'a basın]")
    user_input = input("Haber Başlığı veya Metni Girin (İngilizce): ")
    
    if user_input.lower() == 'q':
        print("Sistem kapatılıyor.")
        break
        
    if len(user_input.split()) < 3:
        print("UYARI: Çok kısa metin girdiniz. Sonuç hatalı olabilir.")

    # Girilen metni vektöre çevir ve tahmin et
    vec = tfidf_vectorizer.transform([user_input])
    pred = pac.predict(vec)
    
    # Sonuç Yazdırma
    if pred[0] == 1:
        print("🔴 SONUÇ: FAKE NEWS (Sahte Haber!)")
    else:
        print("🟢 SONUÇ: REAL NEWS (Gerçek Haber)")