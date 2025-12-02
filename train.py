import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print(">> EĞİTİM BAŞLIYOR...")

# Veri Yükle
try:
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("HATA: CSV dosyaları yok!")
    exit()

df_true['label'] = 0
df_fake['label'] = 1

# Temizlik
df_true['text'] = df_true['text'].apply(lambda x: re.sub(r"^.*?-\s", "", x) if isinstance(x, str) else x)
df_true['combined'] = df_true['title'] + " " + df_true['text']
df_fake['combined'] = df_fake['title'] + " " + df_fake['text']

df = pd.concat([df_true, df_fake]).sample(frac=1).reset_index(drop=True)

# Eğitim
x_train, x_test, y_train, y_test = train_test_split(df['combined'], df['label'], test_size=0.2, random_state=7)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Kaydet
joblib.dump(pac, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print(f"✅ MODEL EĞİTİLDİ VE KAYDEDİLDİ! Başarı: %{accuracy_score(y_test, pac.predict(tfidf_test))*100:.2f}")