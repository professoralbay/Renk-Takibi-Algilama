# Renk-Takibi-Algilama
Bu Python projesi, bir web kamerasından alınan canlı video akışında belirli bir rengi gerçek zamanlı olarak takip eder. OpenCV ve NumPy kütüphanelerini kullanarak, renkli nesnelerin hareketini algılar ve ekranda bir takip çizgisi çizer.
import cv2
import numpy as np

# Takip edilecek rengin HSV (Hue, Saturation, Value) aralığı
# Örnek olarak mavi renk
mavi_alt_sinir = np.array([100, 150, 0])
mavi_ust_sinir = np.array([140, 255, 255])

# Çizgi izini tutmak için bir dizi
iz_cizgisi = []

# Kamera akışını başlat
print("[BILGI] Kamera baslatiliyor...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Kamera acilamadi.")
    exit()

print("[BILGI] Mavi renk takibi basladi, cikmak icin 'q' tusuna basin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gürültüyü azaltmak için bulanıklaştırma
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # Renk uzayını BGR'den HSV'ye dönüştür
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Belirlenen renk aralığındaki pikselleri bul
    mask = cv2.inRange(hsv, mavi_alt_sinir, mavi_ust_sinir)
    
    # Gürültü ve küçük lekeleri temizle
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Maske üzerinde konturları (nesneleri) bul
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    merkez = None
    
    # Eğer en az bir kontur bulunduysa
    if len(contours) > 0:
        # En büyük konturu bul
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) > 500:
            # En büyük konturun etrafına bir daire çiz
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                merkez = (int(x), int(y))
    
    # Merkez noktası bulunduysa, iz çizgisine ekle
    if merkez:
        iz_cizgisi.append(merkez)

    # Çizgiyi çiz
    for i in range(1, len(iz_cizgisi)):
        if iz_cizgisi[i - 1] is None or iz_cizgisi[i] is None:
            continue
        cv2.line(frame, iz_cizgisi[i - 1], iz_cizgisi[i], (0, 0, 255), 2)
        
    cv2.imshow("Canli Renk Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
