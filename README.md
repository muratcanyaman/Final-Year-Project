# Stack Overflow Soru Etiketleme ve Aktif Öğrenme Projesi

Bu proje, Stack Overflow'dan alınan soru verilerini kullanarak, bir soru metnine otomatik olarak uygun etiketleri atayan çok etiketli bir metin sınıflandırma sistemi geliştirmeyi amaçlamaktadır. Proje, öncelikle geleneksel bir makine öğrenmesi yaklaşımıyla (`RandomForest.ipynb`) bir temel model ve performans standardı oluşturur. Ardından, bu temel modelin bulgularını kullanarak, etiketleme maliyetini düşürmek ve model performansını optimize etmek amacıyla aktif öğrenme tekniklerinin (`active_learning.ipynb`) potansiyelini araştırmaktadır.

## Projenin Temel Amaçları

1.  **Otomatik Etiketleme için Temel Model Geliştirme:** Stack Overflow sorularını otomatik olarak doğru teknik etiketlerle sınıflandırabilen, TF-IDF ve Random Forest tabanlı güçlü bir temel sınıflandırma modeli oluşturmak.
2.  **Performans Standardı Belirleme:** Geleneksel yöntemlerle (tüm mevcut etiketli veriyi kullanarak) ulaşılabilecek maksimum performansı belirleyerek aktif öğrenme stratejilerinin etkinliğini ölçmek için bir referans noktası (baseline) sağlamak.
3.  **Aktif Öğrenme ile Verimliliği Artırma:** Entropi tabanlı belirsizlik örneklemesi gibi aktif öğrenme stratejileri kullanarak, temel modele kıyasla daha az etiketlenmiş veri ile benzer veya daha iyi performanslı modeller oluşturmak ve böylece etiketleme eforunu ve maliyetini azaltmak.

## Depo İçeriği ve Yoksayılan Dosyalar

Bu GitHub deposu, projenin ana mantığını içeren Jupyter Notebook'larını (`.ipynb` dosyaları) ve bu `README.md`  dosyasını barındırmak üzere yapılandırılmıştır.

**Aşağıdaki türde dosyalar ve dizinler, boyutları ve yeniden üretilebilirlikleri nedeniyle bu depoya dahil EDİLMEMİŞTİR:**

*   **Ham Veri Dosyaları:** `Orijinal Veri Seti/` dizini altında bulunan `Questions.csv`, `Answers.csv`, `Tags.csv` gibi büyük boyutlu ham veri setleri.
    *   *Neden Yüklenmedi:* Bu dosyalar gigabaytlarca yer kaplamakta ve GitHub repoları için pratik değildir.
    *   *Nasıl Elde Edilir/Kullanılır:* Bu projenin orijinal veri kaynağı Stack Overflow'dur. Veri setinin bir örneği https://www.kaggle.com/datasets/stackoverflow/stacksample/data  üzerinden temin edilebilir. Projeyi kendi verilerinizle çalıştırmak için bu dosyaları `Final-Year-Project/Orijinal Veri Seti/` dizinine yerleştirmeniz gerekmektedir.
*   **İşlenmiş Veri Dosyaları:** `model_ready_sample.csv` gibi, ham verinin işlenmesiyle oluşturulan ve modelleme için kullanılan ara veri dosyaları.
    *   *Neden Yüklenmedi:* Boyutu hala büyük olabilir ve `data_exploration.ipynb` notebook'u çalıştırılarak yeniden üretilebilir.
    *   *Nasıl Üretilir:* `data_exploration.ipynb` notebook'unu çalıştırarak bu dosyayı oluşturabilirsiniz.
*   **Kaydedilmiş Modeller ve TF-IDF Verileri:** `RandomForest_results/Models/` ve `active_learning_results/models/` dizinleri altında bulunan `.joblib` uzantılı eğitilmiş model dosyaları ve TF-IDF vektörleştiricileri.
    *   *Neden Yüklenmedi:* Bu dosyalar da önemli boyutlara ulaşabilir ve ilgili notebook'lar (`RandomForest .ipynb`, `active_learning.ipynb`) çalıştırılarak yeniden eğitilip kaydedilebilir.
    *   *Nasıl Üretilir:* İlgili Jupyter Notebook'ları çalıştırarak modelleri eğitebilir ve kaydedebilirsiniz.

Bu yaklaşım, Git deposunu hafif ve yönetilebilir tutarken, projenin kod mantığının paylaşılmasını ve başkaları tarafından (gerekli veriyi temin ederek ve notebook'ları çalıştırarak) yeniden üretilebilmesini sağlar.

## Önerilen Dizin Yapısı (Yerel Çalışma Ortamı İçin)

Projeyi yerelinizde tam olarak çalıştırmak için aşağıdaki gibi bir dizin yapısı oluşturmanız önerilir (yoksayılan dosyalar dahil):

```
Final-Year-Project/
├── Orijinal Veri Seti/               # Ham veri dosyalarını buraya yerleştirin
│   ├── Questions.csv
│   ├── Answers.csv
│   └── Tags.csv
├── data_exploration.ipynb            # Bu notebook `model_ready_sample.csv`'yi üretir
├── islenmis_veri_analizi.ipynb
├── model_ready_sample.csv            # (data_exploration.ipynb tarafından üretilir)
├── RandomForest .ipynb               # Bu notebook `RandomForest_results/` dizinini üretir
├── RandomForest_results/             # (RandomForest .ipynb tarafından üretilir)
│   ├── Models/
│   └── İmages/
├── active_learning.ipynb             # Bu notebook `active_learning_results/` dizinini üretir
├── active_learning_results/          # (active_learning.ipynb tarafından üretilir)
│   ├── models/
│   ├── images/
│   └── ozet_rapor.txt
├── requirements.txt                  # Gerekli kütüphaneler
└── README.md                         # Bu dosya
```

## Kurulum

1.  **Python Sürümü:** Proje Python 3.12.6 sürümü kullanılarak geliştirilmiş ve test edilmiştir.
2.  **Depoyu Klonlayın (veya indirin):**
    ```bash
    git clone <repository-url>
    cd "Final-Year-Project"
    ```
3.  **Gerekli Kütüphaneleri Yükleyin:**
    Aşağıdaki komutla `requirements.txt` dosyasında listelenen bağımlılıkları yükleyebilirsiniz (öncelikle `requirements.txt` dosyasını kendi ortamınıza göre `pip freeze > requirements.txt` ile güncellemeniz önerilir):
    ```bash
    pip install -r requirements.txt
    ```
    Temel kütüphaneler: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `joblib`, `tqdm`.
4.  **NLTK Veri İndirme (Eğer Gerekliyse):
    `data_exploration.ipynb` notebook'u NLTK kütüphanesinin `wordnet`, `omw-1.4` ve `stopwords` bileşenlerini kullanır. Eğer bunlar sisteminizde yüklü değilse, Python yorumlayıcısında aşağıdaki komutları çalıştırmanız gerekebilir:
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    ```

## Kullanım ve Notebook'lar Arasındaki İlişki

Proje, bir dizi Jupyter Notebook üzerinden yürütülür. Notebook'ların aşağıdaki sırayla çalıştırılması ve aralarındaki mantıksal bağın anlaşılması önemlidir:

1.  **`data_exploration.ipynb`**: Ham veriyi (`Orijinal Veri Seti/` altında olduğu varsayılarak) yükler, temizler, ön işler ve sonraki analizler ile modelleme için temel olacak `model_ready_sample.csv` dosyasını oluşturur.
2.  **`islenmis_veri_analizi.ipynb`**: `model_ready_sample.csv` dosyasını kullanarak işlenmiş verinin detaylı istatistiksel ve görsel analizini yapar.
3.  **`RandomForest .ipynb`**: Bu notebook, `model_ready_sample.csv` üzerinde çalışarak bir Random Forest tabanlı çok etiketli sınıflandırma modeli kurar. Temel amacı, geleneksel bir yaklaşımla ulaşılabilecek en iyi performansı (F1 skoru vb.) belirlemek ve hiperparametreleri optimize etmektir. Çıktıları (`RandomForest_results/` altında) aktif öğrenme aşaması için bir performans referansı ve temel model konfigürasyonu sunar.
4.  **`active_learning.ipynb`**: `RandomForest .ipynb`'de elde edilen bilgiler ışığında ve yine `model_ready_sample.csv` verisini kullanarak aktif öğrenme deneyini yürütür. Random Forest sınıflandırıcısını temel öğrenici olarak kullanır ve entropi tabanlı belirsizlik örneklemesi ile etiketlenecek en değerli verileri akıllıca seçerek modeli iteratif olarak eğitir. Sonuçları (`active_learning_results/` altında) etiketleme verimliliğinin nasıl artırılabileceğini gösterir.

Her bir notebook, kendi içinde detaylı açıklamalar ve adımlar içerir. Çıktı dizinlerinin (`RandomForest_results`, `active_learning_results`) notebook çalıştırılmadan önce var olması gerekebilir veya notebook'lar bunları kendileri oluşturabilir.

## Temel Bulgular

*   **Random Forest Temel Modeli:** Optimize edilmiş Random Forest modeli, soruları birden fazla etiketle sınıflandırmada bir temel performans düzeyi sağlamıştır (tüm veriyle F1 skoru ~0.5233).
*   **Aktif Öğrenmenin Etkinliği:** Aktif öğrenme, daha az etiketli veriyle (%52.9 veri kullanımıyla) rastgele örneklemeye göre daha iyi F1 skorları (%0.6688 vs %0.6574) elde etmiştir. En önemlisi, aktif öğrenme ile eğitilen model, tüm eğitim verisiyle eğitilmiş standart Random Forest modelinden %27.8 daha iyi bir F1 skoru elde etmiştir. Bu, aktif öğrenmenin etiketleme eforunu önemli ölçüde azaltırken model performansını artırma potansiyelini göstermektedir.

## Kullanılan Teknolojiler

*   **Python Sürümü:** 3.12.6
*   **Geliştirme Ortamı:** Jupyter Notebook
*   **Temel Veri Bilimi Kütüphaneleri:** NumPy, Pandas
*   **Makine Öğrenmesi (Scikit-learn):** `TfidfVectorizer`, `RandomForestClassifier`, `MultiOutputClassifier`, `train_test_split`, `precision_recall_fscore_support`, `hamming_loss`, `f1_score`
*   **Doğal Dil İşleme (NLTK):** Stop word kaldırma, Lemmatization (özellikle `data_exploration.ipynb` içinde)
*   **Görselleştirme:** Matplotlib, Seaborn
*   **Model/Veri Serileştirme ve Paralel İşleme (Joblib):** Model kaydı, ara veri kaydı, paralel eğitim/tahmin.
*   **İlerleme Takibi:** TQDM
*   **Diğer Standart Kütüphaneler:** `os`, `time`, `warnings`, `multiprocessing` 
