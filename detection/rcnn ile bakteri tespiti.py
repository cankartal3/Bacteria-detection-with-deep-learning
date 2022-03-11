import cv2
import numpy as np
import random
from scipy import stats
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


    
def bakterisay(filename,r): #dosya adı 
    model = load_model('saved_model.h5')  # cnn modelimizi

    # filename ="55.jpg"
    image = cv2.imread(filename)  # resimi içe aktar
    image = cv2.resize(image,(1000,1000)) # resim boyutubu ayarla #527,240
    # cv2.imshow("Image",image)
    # ilklendir ss  selective search (seçici arama)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)  # tanımlama
    ss.switchToSelectiveSearchQuality()
    
    #print("SS")
    rects = ss.process()  # süreci başlat
    
    proposals = []  # yer önerileri
    boxes = []
    output = image.copy()  # resmi kopyalama (olmasa da olur)
    
    # kaç dükdörtgen için (Burası önemli. Eğer model ezberlemişse bu algoritmayı geliştir.)
    for (x, y, w, h) in rects[:500]: # bu sayıyı değiştir arttır ya da azalt.
        # Önerilen bölgede ki en yüksek ihtilai ya da önerilen bölgede tek gösterim.
        color = [random.randint(0, 255)
                 for j in range(0, 3)]  # rastgele renk üret
        # tespit edilen bölgede dikdörtgen
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        ## algoritmayı kendi planıma göre bu döngüde geliştirecem (maliyeti düşürür)
    
        # region of interest (ilgi bölgesi)
        roi = image[y:y+h, x:x+w]
        # (modelimizi eğittiğimiz boyuta göre
        roi = cv2.resize(roi, dsize=(224, 224),
                         interpolation=cv2.INTER_LANCZOS4)
        # renk skalanını değiştiri      # yeniden boyutlandır,
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        #çünkü modelde derinliği 1         # cv2.INTER_LANCZOS4 resim küçüldüğü için aradaki şeyleri dolduracağımızı ifade eden bir parametre)
        roi = img_to_array(roi)  # resimi arraye çevir
    
        proposals.append(roi)  # ilgili bölgeleri ekle
        boxes.append((x, y, w+x, h+y))  # koordinatları ekle
    
    proposals = np.array(proposals, dtype="float64")    
    boxes = np.array(boxes, dtype="int32")
    
    # print(boxes)
    #print("Sınıflandırma")
    proba = model.predict(proposals)  # yer önerilerini tahmin et
    # print(proba.shape)
    # print("/////////////proba ne////////////////////")
    # print(type(proba))
    list1 = boxes.tolist()
    wanted_box_0 = []
    number_list = []
    idx = []
    for i in range(len(proba)):  # ne kadar tahmin yapıldı
        # en baştan başlayıp en yüksek ihtimali alır
        max_prob = np.max(proba[i, :])
        if max_prob >= 0.99:  # threshold ayarla
            wanted_box_0.append(list1[i])
            idx.append(i)  # olasılık listesi içindeki indexi ekle
            # maximum değerin indexini bul
            number_list.append(np.argmax(proba[i]))
    
    #////////////////////////////////////////////
    wanted_box = []
    for aaa in range(len(idx)):
                    yyy = idx[aaa]
                    if int(np.argmax(proba[yyy]) )==0 or int(np.argmax(proba[yyy])) ==1 or int(np.argmax(proba[yyy])) ==2: # or int(np.argmax(proba[yyy])) ==1 or int(np.argmax(proba[yyy])) ==4 or int(np.argmax(proba[yyy])) ==5:  #or int(np.argmax(proba[yyy])) ==1 or int(np.argmax(proba[yyy])) ==4 or int(np.argmax(proba[yyy])) ==5
                        wanted_box.append(wanted_box_0[aaa])
    #print("list_proba_0: ",len(wanted_box))
    #////////////////////////////////////////////
    
    
    """
    Önerilenlerin sınıflarını ve en çok hangisi sınıflanmış göster.
    """
    wanted_box1 = []
    for qwe in range(len(wanted_box)): # gereksiz boyuttaki kareleri sil ve parametrelerini ayarla
        if int((wanted_box[qwe][2])-(wanted_box[qwe][0])) * int((wanted_box[qwe][3])-(wanted_box[qwe][1])) <22500 and int((wanted_box[qwe][2])-(wanted_box[qwe][0])) * int((wanted_box[qwe][3])-(wanted_box[qwe][1])) >2500:                  
           wanted_box1.append(wanted_box[qwe]) 
    
    
    sayi_say=0
    proba2 = boxes.tolist()
    copyliste1 = wanted_box1.copy()
    asil0 = []
    asil = []
    al_sil = []
    #r = 30
    

    
    #-----------------------------------------------------------------------------------------------------------------------
    """
        Aynı nesnenin üstünde birden fazla dikdörtgen çizmesini engelleyen algoritma.
        
        Seçilen koordinatta r yarıçapında bir daire içinde olan koordinatlardan 1 ranesi rastgele seçilir ve 
    ana resime basılır. Bu dairenin içindeki geriye kalan koordinatlar silinir ve birdaha rastlanmaz bu koordinatlara.
    """
    for i in range (len(wanted_box)):
    
        ekle = wanted_box[i]
        if wanted_box[i][0] !=0 and wanted_box[i][1] != 0 and wanted_box[i][2] != image.shape[0] and wanted_box[i][3] != image.shape[1]:

            for j in range(len(copyliste1)):
                if (ekle[0] - copyliste1[j][0])**2 + (ekle[1] - copyliste1[j][1])**2 <= r**2: # and copyliste1[j][2] - ekle[2] <=50
                    al_sil.append(copyliste1[j])       
            if al_sil != []:
                
                idx0 = []
                list_proba0 = []
                for rrr in range(len(al_sil)):
                    idx0.append(proba2.index(al_sil[rrr]))
                    
                for aa in range(len(idx0)):
                    yy = idx0[aa]
                    list_proba0.append(np.argmax(proba[yy]))
                    # print(np.argmax(proba[yy]))
                    
                      
                m,_ = stats.mode(list_proba0)

                for kkk in range(len(al_sil)):   
                    sayi_say += 1
                    # print("Tahminler:  ",np.argmax(proba[int(proba2.index(al_sil[kkk]))]))
                    if (m[0] == np.argmax(proba[int(proba2.index(al_sil[kkk]))])):
                        asil0.append(al_sil[kkk])
                
                # print('list_proba0:{}\t {} '.format(list_proba0, j)) 
                # print('Mode:',m)
                # print('m nin tipi',type(m))
                # print(m[0])    
                
                rand = random.randint(0, len(asil0)-1)
                asil.append(asil0[rand])
                
                for eee in range(len(asil0)):
                    if (asil0[eee] in copyliste1):
                        copyliste1.remove(asil0[eee])

            al_sil.clear()
            asil0.clear()
            # print(image.shape[1] / (ekle[2] -ekle[0]))

    #print(sayi_say)
    
    # for aaa in range (len(asil)):
    #     print(int(image.shape[1])/int((asil[aaa][2])-(asil[aaa][0])))
    #     print("\t")
    #     print((asil[aaa][3])-(asil[aaa][1]))
    #     print("\n")
        
    proba1 = boxes.tolist()
    
    idx1 = []
    for i in range(len(asil)):
        idx1.append(proba1.index(asil[i]))
    #-----------------------------------------------------------------------------------------------------------------------
    
    color1 = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[125,120,0],]
    
    
    
    
    
    for i in range(len(asil)): # tespit edilen nesnelere dikdörtgen çiz
             
            j = idx1[i]
            if asil[i][0] + asil[i][1] != 0: # RESİMİN KENDİSİNİ TESPİT ETMEMESİ İÇİN ÖNLEM
                cv2.rectangle(image, (asil[i][0], asil[i][1]), (asil[i][2],asil[i][3]),color1[np.argmax(proba[j])],1)
                cv2.putText(image, str(np.argmax(proba[j])),(boxes[j,0] + 5, boxes[j,1] +5), cv2.FONT_HERSHEY_COMPLEX,1.5,color1[np.argmax(proba[j])])
                # cv2.circle(image,(boxes[j,0] + 5, boxes[j,1] +5), 1, (0,255,0), -1)
    # cv2.imshow("winname", image)
    babesia = 0
    plasmodium = 0
    toxoplasma = 0
    for i in range(len(asil)):
        j = idx1[i]
        if str(np.argmax(proba[j]))=="0":
            babesia +=1
        if str(np.argmax(proba[j]))=="1":
            plasmodium +=1
        if str(np.argmax(proba[j]))=="2":
            toxoplasma +=1
            
    return image, babesia, plasmodium, toxoplasma #(len(asil)) # tespit edilenlerin olduğu resimi ve tespit sayısını döndür




# #-----------------------TEST--------------------------------
# dosya = "hepsi2.jpg"        
# bakterisay(dosya)
# #-----------------------TEST--------------------------------   
       