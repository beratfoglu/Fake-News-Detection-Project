# 🕵️‍♂️ Fake News Detection System (Sahte Haber Tespit Sistemi)

Bu proje, Makine Öğrenmesi (Machine Learning) ve Doğal Dil İşleme (NLP) teknikleri kullanılarak haber metinlerinin gerçek mi yoksa sahte mi olduğunu tespit eden bir yapay zeka uygulamasıdır.

## 🚀 Proje Hakkında
Günümüzde bilgi kirliliği ve dezenformasyonun artmasıyla birlikte, haberlerin doğruluğunu teyit etmek zorlaşmıştır. Bu proje, **Passive Aggressive Classifier** algoritmasını kullanarak haber metinleri üzerinde **%90 üzeri doğruluk oranıyla** sınıflandırma yapar.

### 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **Scikit-learn:** Makine öğrenmesi modeli ve vektörleştirme (TF-IDF) için.
* **Pandas & Numpy:** Veri manipülasyonu ve analizi için.

## 📂 Veri Seti
Proje, Kaggle üzerinden sağlanan açık kaynaklı "Fake and Real News Dataset" kullanılarak eğitilmiştir. Veri seti, binlerce politik ve güncel haber metnini içerir.

## 📸 Ekran Görüntüleri ve Test Sonuçları

### 1. Model Başarı Oranı
Model eğitildikten sonra elde edilen doğruluk skoru:
![Başarı Oranı](basari_orani.png)

### 2. Gerçek Haber Testi (Real News)
BBC kaynağından alınan veri ile yapılan test sonucu:
![Gerçek Haber Testi](test_real.png)

### 3. Sahte Haber Testi (Fake News)
Manipülatif içerik ile yapılan test sonucu:
![Sahte Haber Testi](test_fake.png)

## 💻 Kurulum ve Çalıştırma

1. Projeyi bilgisayarınıza klonlayın veya indirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt