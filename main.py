import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
# --- YENİ KÜTÜPHANE ---
from deep_translator import GoogleTranslator 
# ----------------------

print("---------------------------------------------------------")
print("PROJE BAŞLATILIYOR... (Multilingual Version v2.0)")
print("---------------------------------------------------------")

# --- 1. VERİLERİ YÜKLEME ---
try:
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("HATA: Dosyalar bulunamadı!")
    exit()

df_true['label'] = 0
df_fake['label'] = 1

# Reuters temizliği
try:
    df_true['text'] = df_true['text'].apply(lambda x: re.sub(r"^.*?-\s", "", x) if isinstance(x, str) else x)
except:
    pass

df_true['combined_text'] = df_true['title'] + " " + df_true['text']
df_fake['combined_text'] = df_fake['title'] + " " + df_fake['text']

dataframe = pd.concat([df_true, df_fake]).sample(frac=1).reset_index(drop=True)

# --- MODEL EĞİTİMİ ---
print(">> Model eğitiliyor...")
x = dataframe['combined_text']
y = dataframe['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

basari = round(accuracy_score(y_test, pac.predict(tfidf_test)) * 100, 2)
print(f"✅ MODEL GÜNCELLENDİ! Başarı Oranı: %{basari}")

# --- 7. TEST (ÇEVİRİ ENTEGRASYONLU) ---
print("\n" + "="*60)
print("🌍 ÇOKLU DİL DESTEĞİ AKTİF (Google Translate Altyapısı)")
print("Türkçe, Almanca, Fransızca... İstediğin dilde haber girebilirsin.")
print("="*60)

# Çevirmen nesnesini oluştur (Otomatik algıla -> İngilizceye çevir)
translator = GoogleTranslator(source='auto', target='en')

while True:
    print("\n[Çıkmak için 'q' yazıp Enter'a basın]")
    user_input = input("Haber Metni Girin: ")
    
    if user_input.lower() == 'q':
        break
        
    if len(user_input.split()) < 3:
        print("UYARI: Çok kısa metin girdiniz. Model yanılabilir!")
        # Kısa metin girilirse devam et ama uyar
    
    try:
        # 1. ADIM: Çeviri Yap
        print(">> Dil algılanıyor ve İngilizceye çevriliyor...")
        translated_text = translator.translate(user_input)
        
        # Kullanıcıya çeviriyi gösterelim ki "Neden Fake dedi?" diye şaşırmasın
        print(f"   📝 Çevrilmiş Hali: \"{translated_text}\"")
        
        # 2. ADIM: Tahmin Et
        vec = tfidf_vectorizer.transform([translated_text])
        pred = pac.predict(vec)
        
        if pred[0] == 1:
            print("🔴 SONUÇ: FAKE NEWS (Sahte Haber)")
        else:
            print("🟢 SONUÇ: REAL NEWS (Gerçek Haber)")
            
    except Exception as e:
        print(f"Bağlantı Hatası: {e}")