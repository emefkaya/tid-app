import cv2
import mediapipe as mp

# MediaPipe kurulumları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Parmak uçları ve orta eklemlerin MediaPipe ID'leri
# Sırasıyla: İşaret, Orta, Yüzük, Serçe parmak
parmak_uclari = [8, 12, 16, 20]
parmak_eklemleri = [6, 10, 14, 18]

cap = cv2.VideoCapture(0)

print("Kamera açılıyor... Çıkmak için 'q' tuşuna basın.")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Görüntüyü çevir (Ayna efekti, daha rahat kullanım için)
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            
            # Hangi parmakların açık olduğunu tutacağımız liste
            acik_parmaklar = []
            
            # 4 parmak için döngü (Başparmak hariç, onun açısı farklıdır)
            for id in range(4):
                # Eğer parmak ucu, eklemden daha yukarıdaysa (Y değeri daha küçükse)
                if handLms.landmark[parmak_uclari[id]].y < handLms.landmark[parmak_eklemleri[id]].y:
                    acik_parmaklar.append(1) # Parmak açık
                else:
                    acik_parmaklar.append(0) # Parmak kapalı
            
            # Hareket Algılama Mantığı
            toplam_acik = acik_parmaklar.count(1)
            
            hareket_metni = "Tanimlanmadi"
            if toplam_acik == 4:
                hareket_metni = "Acik El (Dur)"
            elif toplam_acik == 0:
                hareket_metni = "Yumruk"
            elif toplam_acik == 2 and acik_parmaklar[0] == 1 and acik_parmaklar[1] == 1:
                hareket_metni = "Baris Isareti (V)"
                
            # Algılanan hareketi ekrana yazdır
            cv2.putText(img, hareket_metni, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Isaret Dili Demo", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()