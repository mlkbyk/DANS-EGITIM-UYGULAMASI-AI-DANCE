🕺 AI Destekli Gerçek Zamanlı Hip-Hop Dans Değerlendirme Sistemi
Bu proje, hip-hop dans performanslarını gerçek zamanlı olarak analiz edip değerlendiren bir yapay zeka uygulamasıdır. Kullanıcıdan alınan canlı video görüntüsü, MediaPipe Pose ile poz verilerine çevrilir; bu veriler referans danslarla karşılaştırılarak Gemini API üzerinden kullanıcıya kişiselleştirilmiş geri bildirim sağlanır. Ayrıca bir dans eğitmeni chatbot, öğrenme sürecine rehberlik eder.

🧠 Özellikler
✅ Gerçek zamanlı kamera görüntüsünden poz analizi
✅ Eklem açıları üzerinden dans benzerlik puanı hesaplama
✅ LangChain + Gemini API ile doğal dil destekli geri bildirim üretimi
✅ Kullanıcının dans stiline göre özelleştirilmiş anlık tavsiyeler
✅ Chatbot + RAG sistemi ile dans teknikleri hakkında bilgi verme
✅ React frontend, Flask backend, JSON veri işleme altyapısı
✅ Referans JSON dans veri kümesinden değerlendirme (ideal pozlar)
✅ Anlık hata tespiti: Hangi açı ne kadar sapmış, neden başarısız
✅ Gelişmiş müzik senkronizasyon kontrolleri
✅ Fare izi efekti, ekran bölmeli dans takibi (ref. vs user)

🧩 Kullanılan Teknolojiler
🎛️ Frontend:
Teknoloji	Açıklama
React.js	Kullanıcı arayüzü ve bileşen yönetimi
MediaPipe Pose	Google MediaPipe ile gerçek zamanlı insan poz tahmini
@mediapipe/camera_utils	Webcam'den canlı video alma ve işleme
Canvas API	Eklem bağlantılarını ve koordinatları görsel olarak çizme
React Hooks (useRef, useState, useEffect)	Kamera ve poz verisi yönetimi
fetch API	Flask backend ile veri alışverişi

🧠 Backend:
Teknoloji	Açıklama
Flask	REST API ile frontend-backend iletişimi
LangChain	LLM (Large Language Model) isteklerini düzenleme ve bağlam yönetimi
Gemini 1.5 Flash API	Google’ın yeni nesil LLM’i ile hızlı geri bildirim üretimi
dotenv	API anahtarlarının güvenli şekilde saklanması
Rate limiter	Gemini API çağrılarını sınırlamak için koruma katmanı

🔢 Veri İşleme:
Katman	Açıklama
Pose landmark JSON verisi	Kullanıcının her frame'deki poz verileri (33 anahtar nokta)
Eklem açıları hesaplama	calculate_angle() fonksiyonu ile her eklemin trigonometri kullanılarak derecesel açısı hesaplanır
Benzerlik puanı üretimi	Kullanıcı ile referans JSON arasındaki açısal farkların ortalaması alınır
Sapma toleransı kontrolü	Belirli açılar 10-15° tolerans ile değerlendirilebilir

🤖 Yapay Zeka & RAG:
Bileşen	Açıklama
LangChain Retriever	Dans tekniklerini içeren metinsel veri tabanına dayalı arama yapar
Google Generative AI Embeddings	Metinleri vektörleştirerek Chroma veri tabanına entegre eder
Gemini Chat Prompt	Kullanıcının mesajlarını dans eğitmeni gibi analiz eder, yanıtlar
RAG (Retrieval-Augmented Generation)	Chat'e bağlamlı, bilgi tabanlı yanıtlar verir

🎬 Kullanım Akışı
Kullanıcı sağ ekranda referans videoyu izler, sol ekranda kamera aktiftir.

MediaPipe Pose, kameradan alınan görüntüleri frame frame işler.

Elde edilen poz landmark'ları, eklem açılarına çevrilir ve Flask API’ye gönderilir.

API, kullanıcı ile referans JSON dans verisini karşılaştırır.

Hatalı açı varsa kullanıcıya düzeltici uyarı gösterilir.

Kullanıcı, chatbot’a "kolum neden doğru değil?" gibi doğal dille soru sorabilir.

Gemini modeli hem açıklama yapar hem de cesaretlendirici bir dille motivasyon sağlar.
