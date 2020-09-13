#================================================================
#Author=mete => meteserce2@gmail.com
#================================================================ 
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations


def is_close(p1, p2):
    """
    #================================================================
    # 1. Amaç: İki nokta arasındaki Öklid Mesafesini Hesaplayın
    #================================================================
    :p1, p2 = Öklid Mesafesini hesaplamak için iki nokta

     :dönüş:
     dst = İki 2d nokta arasındaki Öklid Mesafesi
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 


def convertBack(x, y, w, h): 
    #================================================================
    # 2. Amaç: Merkez koordinatlarını dikdörtgen koordinatlara dönüştürür
    #================================================================
    """
     param:
     x, y = bbox'ın orta noktası
     w, h = bbox'ın genişliği, yüksekliği

     :dönüş:
     xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    :param:
    algılamalar = bir çerçevede toplam algılama
     img = darknet'in Detection_image yönteminden görüntü
     :dönüş:
     bbox ile img
    """
    #================================================================
    #3.1 Amaç: Kişiler sınıfını tespitlerden filtreleyin ve
     # Her kişi tespiti için sınırlayıcı kutu centroid.
    #================================================================
    if len(detections) > 0:  						# Görüntüde en az 1 algılama ve bir çerçevede algılama varlığını kontrol etme
        centroid_dict = dict() 						# İşlev bir sözlük oluşturur ve buna centroid_dict adını verir
        objectId = 0								# Object Id adlı bir değişkeni başlatıp 0 olarak ayarlıyoruz
        for detection in detections:				# Bu if ifadesinde, tüm algılamaları yalnızca kişiler için filtreliyoruz
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   # Coco dosyasında tüm isimler dizesi var
            if name_tag == 'person':                
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]      	# Algılamaların merkez noktalarını saklayın
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Merkez koordinatlardan dikdörtgen koordinatlara dönüştürün, BBox'un hassasiyetini sağlamak için yüzer kullanıyoruz
                # Append center point of bbox for persons detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Dizin merkez noktaları ve bbox olarak 'objectId' ile tuple sözlüğü oluşturun
                objectId += 1 #Her algılama için dizini artırın
    #=================================================================#
    
    #=================================================================
    #3.2 Amaç: Hangi bbox'ın birbirine yakın olduğunu belirleyin
    #=================================================================            	
        red_zone_list = [] # Eşik altında mesafe durumunda hangi Nesne kimliğinin bulunduğunu içeren liste.
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Yakın tespitlerin tüm kombinasyonlarını alın, # Birden çok öğe listesi - id 1 1, punto 2, 1,3
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Centroid x: 0, y: 1 arasındaki farkı kontrol edin
            distance = is_close(dx, dy) 			# CÖklid mesafesini hesaplar
            if distance < 75.0:						# Sosyal mesafe eşiğimizi ayarlayın - Bu koşulu karşılıyorlarsa ..
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  Listeye kimlik ekle
                    red_line_list.append(p1[0:2])   #  Listeye puan ekleyin
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		# İkinci kimlik için aynı
                    red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  # dict (1 (anahtar): kırmızı (değer), 2 mavi) idx - anahtar kutusu - değer
            if idx in red_zone_list:   # id kırmızı bölge listesindeyse
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Kırmızı sınırlayıcı kutular oluşturun # başlangıç noktası, bitiş noktası boyutu 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Yeşil sınırlayıcı kutular oluşturun
		#=================================================================#

		#=================================================================
    	# 3.3 Amaç: Risk Analizini Görüntüleme ve Risk Göstergelerini Gösterme
    	#=================================================================        
        text = "Sosyal mesafeyi ihlal edenler: %s" % str(len(red_zone_list)) 			 #Risk Altındaki İnsanları Sayma
        location = (10,25)												# Görüntülenen metnin konumunu ayarlayın
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,76,76), 2, cv2.LINE_AA)  # Metni Görüntüle

            for check in range(0, len(red_line_list)-1):					# Yakındaki kutular arasında çizgi çizin kırmızı liste öğeleri arasında yinelenir
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# X için çizgi koordinatlarını hesaplayın
            check_line_y = abs(end_point[1] - start_point[1])			# Y için çizgi koordinatlarını hesaplayın
            if (check_line_x < 75) and (check_line_y < 25):				# Her ikisi de varsa çizgilerin eşik mesafemizin altında olup olmadığını kontrol ederiz.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Yalnızca eşik değerinin üzerindeki çizgiler görüntülenir.
        #=================================================================#
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Nesne algılama gerçekleştirin
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg" #Kullanacağımız yolo dosyaları ve dosya yolları
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cv2.namedWindow("mete",cv2.WINDOW_NORMAL)
    #cap = cv2.VideoCapture("udp://@0.0.0.0:11111") # "udp://@0.0.0.0:11111" or 0 tello drone ile görüntü almak yada pc webcam için kendinize göre ayarlayın bunu kullanacaksanız alttaji satırı yoruma alın
    cap = cv2.VideoCapture("sosyaltest.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
            "test5_output4.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Her tespit için yeniden kullandığımız bir görüntü oluşturun
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Çerçeve mevcut mu kontrol edin :: 'ret', çerçeve varsa True döndürür, aksi takdirde döngüyü kırın
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('mete', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
