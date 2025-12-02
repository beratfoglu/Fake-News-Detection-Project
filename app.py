from flask import Flask, render_template, request, jsonify
import joblib
import os
from deep_translator import GoogleTranslator
from newspaper import Article, Config # <-- Config eklendi
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# Modelleri Yükle
if os.path.exists("fake_news_model.pkl"):
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
else:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model bulunamadı! Önce train.py çalıştırın.'})

    try:
        data = request.get_json()
        content_input = data.get('content') # URL veya Metin
        
        text_to_process = ""

        # --- 1. DURUM: URL GELDİYSE ---
        if data.get('type') == 'url':
            try:
                # Siteye kendimizi Chrome gibi tanıtıyoruz (Engellenmemek için)
                user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                
                config = Config()
                config.browser_user_agent = user_agent
                config.request_timeout = 10

                article = Article(content_input, config=config)
                article.download()
                article.parse()
                
                # --- TEMİZLİK KISMI BURADA ---
                ham_metin = article.text
                
                # BBC vb. sitelerdeki gereksiz buton yazılarını siliyoruz
                temiz_metin = ham_metin.replace("Share", "").replace("Save", "").replace("Subscribe", "")
                
                # Satır boşluklarını ve kaymaları düzeltiyoruz
                text_to_process = " ".join(temiz_metin.split())
                # -----------------------------
                
                if len(text_to_process) < 50:
                     return jsonify({'error': 'Metin çekilemedi veya çok kısa.'})

            except Exception as e:
                return jsonify({'error': f'Link hatası: {str(e)}'})
        
        # --- 2. DURUM: METİN GELDİYSE ---
        else:
            text_to_process = content_input

        # Çeviri (Google Translate)
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text_to_process[:2000]) # İlk 2000 karakter
        
        # Tahmin
        vec = vectorizer.transform([translated])
        pred = model.predict(vec)[0]

        return jsonify({'isFake': bool(pred == 1), 'translation': translated})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)