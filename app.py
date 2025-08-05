from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import math
from dotenv import load_dotenv

# Langchain kütüphaneleri
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# .env dosyasını yükle
load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv("API_KEY")
if not api_key:
    print("HATA: API_KEY ortam değişkeni ayarlanmamış. Lütfen .env dosyasını kontrol edin.")
    raise ValueError("API anahtarı .env dosyasından yüklenemedi.")
else:
    genai.configure(api_key=api_key)

# Langchain için Gemini modelini ve embeddings'i başlat
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Dans verisi
hiphop_dance_data = [
    {
        "name": "Toprock",
        "category": "Footwork",
        "style": "Breaking",
        "description": "Breakdance'in ayakta yapılan açılış hareketleri.",
        "tips": ["Dengeyi koru", "Ayakları ritme uygun hareket ettir", "Ellerle groove'u destekle"],
        "difficulty": "Orta",
        "common_mistakes": ["Karmaşık adım dizilimi", "Denge kaybı", "Zayıf kol hareketleri"],
        "evaluation_criteria": ["Rhythm Match", "Foot Precision", "Balance"]
    },
    {
        "name": "Six-Step",
        "category": "Footwork",
        "style": "Breaking",
        "description": "Yerde yapılan temel 6 adımlı breaking hareketi.",
        "tips": ["Adımları temiz uygula", "Ellerin pozisyonunu koru", "Vücut ağırlığını iyi dağıt"],
        "difficulty": "Zor",
        "common_mistakes": ["Adım sayısının karıştırılması", "Hızdan dolayı kontrol kaybı"],
        "evaluation_criteria": ["Timing", "Smooth Transition", "Floor Control"]
    },
    # ... Diğer dans hareketleri aynı şekilde devam eder
]

# RAG sistemi için metin hazırlığı
rag_source_texts = []
for dance_move in hiphop_dance_data:
    text_chunk = (
        f"Dans Hareketi Adı: {dance_move['name']}\n"
        f"Kategori: {dance_move['category']}\n"
        f"Açıklama: {dance_move['description']}\n"
        f"İpuçları: {', '.join(dance_move['tips'])}\n"
        f"Sık Yapılan Hatalar: {', '.join(dance_move['common_mistakes'])}\n"
        f"Zorluk: {dance_move['difficulty']}\n"
        f"Değerlendirme Kriterleri: {', '.join(dance_move['evaluation_criteria'])}\n"
    )
    rag_source_texts.append(text_chunk)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents(rag_source_texts)

vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Langchain zincirleri ve promptlar
chat_prompt_template = PromptTemplate(
    input_variables=["user_message"],
    template="""
Bir dans eğitmenisin ve bir öğrenciyle konuşuyorsun. Bu öğrencinin mesajı: '{user_message}'.
Ona kısa, motive edici ve en fazla 2 cümle olacak şekilde cevap ver.
"""
)
chat_chain = LLMChain(llm=llm, prompt=chat_prompt_template)

pose_prompt_template = PromptTemplate(
    input_variables=["most_problematic_joint", "user_angle", "reference_angle"],
    template="""
Sen, bir dans eğitmenisin. Öğrencinin ve referansın {most_problematic_joint} açısı aşağıdaki gibidir.
Bu açıdaki farka odaklanarak, tek bir net ve motive edici düzeltme önerisi sun.
Yanıtın sadece 1-2 cümle uzunluğunda olsun.

Kullanıcı {most_problematic_joint} Açısı: {user_angle:.2f} derece
Referans {most_problematic_joint} Açısı: {reference_angle:.2f} derece
"""
)
pose_eval_chain = LLMChain(llm=llm, prompt=pose_prompt_template)

rag_prompt_template = PromptTemplate(
    template="""
Sen bir dans eğitmenisin. Aşağıdaki bağlamı kullanarak kullanıcının sorusuna kısa ve anlaşılır bir yanıt ver.
Eğer bağlamda soruya net bir cevap yoksa, "Bu konuda bir bilgim yok." şeklinde yanıt ver.

---
Bağlam:
{context}
---
Soru: {input}
""",
    input_variables=["context", "input"]
)
rag_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, rag_prompt_template)
)

# Fonksiyon: açı hesaplama
def calculate_angle(p1, p2, p3):
    """Üç eklem noktasının arasındaki açıyı hesaplar."""
    if not p1 or not p2 or not p3:
        return None
    try:
        v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'])
        v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return None
        angle = math.acos(min(max(dot_product / (magnitude_v1 * magnitude_v2), -1.0), 1.0))
        return math.degrees(angle)
    except (KeyError, TypeError):
        return None

# Flask endpointleri
@app.route('/')
def home():
    return "Flask backend sunucusu başarıyla çalışıyor!", 200

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "Message field is required"}), 400
    try:
        response = chat_chain.invoke({"user_message": user_message})
        return jsonify({"response": response['text']})
    except Exception as e:
        print(f"Gemini API hatası (chat): {e}")
        return jsonify({"error": "Chatbot yanıtı alınamadı."}), 500

@app.route('/evaluate_pose', methods=['POST'])
def evaluate_pose():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.json
    user_pose = data.get('user_pose')
    reference_pose = data.get('reference_pose')
    if not user_pose or not reference_pose or len(user_pose) < 33 or len(reference_pose) < 33:
        return jsonify({'error': 'Eksik veya geçersiz poz verisi'}), 400

    user_angles = {
        'sağ_dirsek': calculate_angle(user_pose[11], user_pose[13], user_pose[15]),
        'sol_dirsek': calculate_angle(user_pose[12], user_pose[14], user_pose[16]),
        'sağ_omuz': calculate_angle(user_pose[13], user_pose[11], user_pose[23]),
        'sol_omuz': calculate_angle(user_pose[14], user_pose[12], user_pose[24]),
        'sağ_diz': calculate_angle(user_pose[23], user_pose[25], user_pose[27]),
        'sol_diz': calculate_angle(user_pose[24], user_pose[26], user_pose[28]),
        'sağ_kalça': calculate_angle(user_pose[11], user_pose[23], user_pose[25]),
        'sol_kalça': calculate_angle(user_pose[12], user_pose[24], user_pose[26])
    }
    reference_angles = {
        'sağ_dirsek': calculate_angle(reference_pose[11], reference_pose[13], reference_pose[15]),
        'sol_dirsek': calculate_angle(reference_pose[12], reference_pose[14], reference_pose[16]),
        'sağ_omuz': calculate_angle(reference_pose[13], reference_pose[11], reference_pose[23]),
        'sol_omuz': calculate_angle(reference_pose[14], reference_pose[12], reference_pose[24]),
        'sağ_diz': calculate_angle(reference_pose[23], reference_pose[25], reference_pose[27]),
        'sol_diz': calculate_angle(reference_pose[24], reference_pose[26], reference_pose[28]),
        'sağ_kalça': calculate_angle(reference_pose[11], reference_pose[23], reference_pose[25]),
        'sol_kalça': calculate_angle(reference_pose[12], reference_pose[24], reference_pose[26])
    }

    angle_diffs = {
        key: abs(user_angles[key] - reference_angles[key])
        for key in user_angles.keys()
        if user_angles.get(key) is not None and reference_angles.get(key) is not None
    }

    if all(diff < 5 for diff in angle_diffs.values()):
        return jsonify({'feedback': "Mükemmel! Pozisyonu harika yakaladın, hiçbir düzeltmeye ihtiyacın yok. Harika iş!"})

    most_problematic_joint = max(angle_diffs, key=angle_diffs.get)
    try:
        response = pose_eval_chain.invoke({
            "most_problematic_joint": most_problematic_joint,
            "user_angle": user_angles[most_problematic_joint],
            "reference_angle": reference_angles[most_problematic_joint]
        })
        return jsonify({'feedback': response['text']})
    except Exception as e:
        print(f"Gemini API hatası (evaluate_pose): {e}")
        return jsonify({'error': "Gemini API'den geri bildirim alınırken bir hata oluştu."}), 500

@app.route('/rag_query', methods=['POST'])
def rag_query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query field is required"}), 400
    try:
        response = rag_chain.invoke({"input": user_query})
        return jsonify({"response": response['answer']})
    except Exception as e:
        print(f"RAG sorgusu hatası: {e}")
        return jsonify({"error": "Sorgu işlenirken bir hata oluştu."}), 500

if __name__ == '__main__':
    print("Flask backend sunucusu başlatılıyor...")
    app.run(debug=True, port=5000)
