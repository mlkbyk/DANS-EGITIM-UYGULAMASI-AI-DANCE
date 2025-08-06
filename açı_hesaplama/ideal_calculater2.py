import json
import os
import numpy as np
import mediapipe as mp  


mp_pose = mp.solutions.pose


reference_data_directory = r"C:\Users\MS\dance-tracker\src\reference_data"


output_directory = r"C:\Users\MS\dance-tracker\src\reference_data"
os.makedirs(output_directory, exist_ok=True)  


json_extension = '_pose_data.json'


WINDOW_SIZE = 25 



IMPORTANT_ANGLES = [
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "left_hip_angle", "right_hip_angle",
    "left_knee_angle", "right_knee_angle",
    # "left_neck_angle", "right_neck_angle" # İsteğe bağlı, boyun hareketleri için
]


IMPORTANT_LANDMARK_IDS = list(set([
    mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    mp_pose.PoseLandmark.NOSE.value, mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value
    # Boyun için
]))


VISIBILITY_PENALTY_FACTOR = len(IMPORTANT_ANGLES) * 50  

print(f"Referans veri dizini: {reference_data_directory}")
print(f"Çıktı dizini: {output_directory}")
print(f"Pencere boyutu (kare): {WINDOW_SIZE}")
print(f"Görünürlük ceza faktörü: {VISIBILITY_PENALTY_FACTOR}")


for filename in os.listdir(reference_data_directory):
    if filename.endswith(json_extension):
        json_path = os.path.join(reference_data_directory, filename)
        video_name = filename.replace(json_extension, '')  

        print(f"\n--- '{filename}' dosyası işleniyor... ---")

        try:
            with open(json_path, 'r') as f:
                frame_data = json.load(f)  
        except json.JSONDecodeError as e:
            print(f"Hata: '{filename}' dosyası okunurken JSON hatası oluştu: {e}")
            continue
        except FileNotFoundError:
            print(f"Hata: '{filename}' dosyası bulunamadı.")
            continue

        if not frame_data:
            print(f"Uyarı: '{filename}' dosyasında hiç kare verisi yok. Atlanıyor.")
            continue
        if len(frame_data) < WINDOW_SIZE:
            print(
                f"Uyarı: '{filename}' dosyasındaki kare sayısı ({len(frame_data)}) pencere boyutundan ({WINDOW_SIZE}) küçük. Atlanıyor.")
            continue

        
        frame_quality_scores = []
        for i in range(len(frame_data) - 1):
            current_frame = frame_data[i]
            next_frame = frame_data[i + 1]

            current_frame_angles = current_frame['angles']
            next_frame_angles = next_frame['angles']
            current_frame_landmarks = current_frame['landmarks']

            
            frame_delta_sum = 0
            for angle_name in IMPORTANT_ANGLES:
                if angle_name in current_frame_angles and angle_name in next_frame_angles:
                    delta = abs(current_frame_angles[angle_name] - next_frame_angles[angle_name])
                    frame_delta_sum += delta

            
            total_visibility = 0
            visible_landmark_count = 0
            for lm_data in current_frame_landmarks:
                if lm_data['id'] in IMPORTANT_LANDMARK_IDS:
                    total_visibility += lm_data['visibility']
                    visible_landmark_count += 1

            avg_visibility = total_visibility / visible_landmark_count if visible_landmark_count > 0 else 0

           
            visibility_penalty = (1 - avg_visibility) * VISIBILITY_PENALTY_FACTOR

            # Kare kalitesi skoru: Açı değişimi + Görünürlük cezası
            frame_quality_scores.append(frame_delta_sum + visibility_penalty)

        # En istikrarlı ve kaliteli (en düşük toplam skora sahip) pencereyi bul
        min_window_quality_score = float('inf')
        best_start_frame_index = -1

        # Pencereyi kaydırarak en iyi segmenti bul
        # frame_quality_scores listesi frame_data'dan 1 kısa olduğu için döngü sınırına dikkat et
        for i in range(len(frame_quality_scores) - WINDOW_SIZE + 1):
            current_window_quality_sum = sum(frame_quality_scores[i: i + WINDOW_SIZE])

            if current_window_quality_sum < min_window_quality_score:
                min_window_quality_score = current_window_quality_sum
                best_start_frame_index = i

        if best_start_frame_index == -1:
            print(f"Uyarı: '{filename}' için ideal segment bulunamadı. Atlanıyor.")
            continue

        # İdeal segmenti çıkar
        
        ideal_segment_frames = frame_data[best_start_frame_index: best_start_frame_index + WINDOW_SIZE]

        # Çıktı JSON dosyasının adı
        output_json_filename = f"{video_name}_ideal_segment.json"
        output_path = os.path.join(output_directory, output_json_filename)

        # İdeal segmenti yeni bir JSON dosyasına kaydet
        with open(output_path, 'w') as f:
            json.dump(ideal_segment_frames, f, indent=4)

        print(f"'{filename}' için ideal segment başarıyla kaydedildi: {output_json_filename}")
        print(f"Başlangıç karesi (yaklaşık): {best_start_frame_index + 1}")  # +1 çünkü liste 0'dan başlar
        print(f"Toplam pencere kalite skoru: {min_window_quality_score:.2f}")

print("\nTüm JSON dosyalarının işlenmesi tamamlandı!")
