from tensorflow.keras.models import load_model
import os 
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical #
from sklearn.model_selection import train_test_split #veriyi 2'ye ayırıyor train ve test
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # farklı resimler generate (oluşturmak) eder
from sklearn import metrics
from tedcombine import tedc
""""
İlk önce data seti yükle ve eğitim,test ve doğrulama olarak ayır.
Kayıtlı modeli içe aktar ve metrikleri hesapla.

"""

def data_set_ayir_preProcess(yol):

    
    myList = os.listdir(yol) # path içindeki klasörleri myLiset değişkeninde atadık
    noOfClasses = len(myList) # klasör sayısı
    
    print("Label(sınıf) sayısı: ",noOfClasses)
    
    images = []
    classNo = []
    
    for i in range(noOfClasses): #klasörlerin içinde dolaşmak için
        myImageList = os.listdir(yol + "\\"+myList[i]) 
        for j in myImageList:
            
            img = cv2.imread(yol + "\\" + myList[i] + "\\" + j)
            img = cv2.resize(img, (224,224)) # modelin hızlı olabilmesi için resimleri yeniden boyutlandırıp boyutlarını küçülttük.
            img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img) # resimleri image listesine attık
            classNo.append(i) # etiketleri classNo listesine attık
            
    # print(len(images))
    # print(len(classNo))
    
    images = np.array(images)
    classNo = np.array(classNo)
    
    
    print(images.shape)
    print(classNo.shape)
    
    # veri ayırma
    x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.2, random_state = 42)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)
    
    # verimizi 2'ye böldük train ve test olarak. y_test saklıcaz ve verimiz en son ana kadar görmicek. 
    # veriy ieğitirken validation (doğrulama) yapmamız gerekiyor. Veriyi x_train, x_validation veri setlerini kullanarak eğiticez
    # Veri hazır olduğunda y_test veri setini kullanarak doğrulama yapıcaz
    
    
    print("\nVeri seti: ")  #veri setlerinin adatleri ve boyutları
    print(images.shape)
    print("\nEğitim: ")
    print(x_train.shape)
    print("\nTest: ")
    print(x_test.shape)
    print("\nDoğrulama: ")
    print(x_validation.shape)
    
    # # Dağılımları görselleştirme
    # fig, axes = plt.subplots(3,1,figsize=(7,7)) 
    # fig.subplots_adjust(hspace = 0.5)
    # sns.countplot(y_train, ax = axes[0])
    # axes[0].set_title("y_train")
    
    # sns.countplot(y_test, ax = axes[1])
    # axes[1].set_title("y_test")
    
    # sns.countplot(y_validation, ax = axes[2])
    # axes[2].set_title("y_validation")
    
    
    # # preprocess 
    # def preProcess(img):
    #     # img = tedc(img)  # her resim için tek tek arkaplan çıkarma
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # siyah beyaza çevirdik
    #     img = cv2.equalizeHist(img) # histogramı 0 ile 255 arasına kadar genişlettik
    #     img = img /255 # normalize ettik
        
    #     return img
    
    # # idx = 10000
    # # img = preProcess(x_train[idx])
    # # img = cv2.resize(img,(300,300))
    # # cv2.imshow("Preprocess ",img)
    
        
    # # # preprocess işlemini tüm verimize uyguluyoruz map yöntemi ile
    # x_train = np.array(list(map(preProcess, x_train))) # map(fonk, parametreler)
    # x_test = np.array(list(map(preProcess, x_test)))
    # x_validation = np.array(list(map(preProcess, x_validation)))
    
    x_train = x_train.reshape(-1,224,224,3)  # -1 yaptığımız yeri kod ne ayarlarsa o olsun istiyoz. 300 resim varsa burası 300 olucak.
    # print(x_train.shape)
    x_test = x_test.reshape(-1,224,224,3)
    x_validation = x_validation.reshape(-1,224,224,3)
    
    # data generate (oluşturmak). Farklı açılar ve uzaklıklar oluşturuyoz data setin farklılık açısından zenginleşmesi için
    dataGen = ImageDataGenerator(
                                  width_shift_range = 0.1,  #kaydırma
                                  height_shift_range = 0.1,
                                  zoom_range = 0.1,  # zoom
                                  rotation_range = 10
                                  
                                  # rotation_range=15,
                                  # rescale=1./255,
                                  # shear_range=0.1,
                                  # zoom_range=0.2,
                                  # horizontal_flip=True,
                                  # width_shift_range=0.1,
                                  # height_shift_range=0.1
                                  ) 
    
    
    
    dataGen.fit(x_train) # bu generate işlemini train için uyguladık. Çünkü eğitimde farklılığı arttırmak istedik

    
    y_train = to_categorical(y_train, noOfClasses)  # keras için yapılması gerekiyor
    y_test = to_categorical(y_test, noOfClasses)
    y_validation = to_categorical(y_validation, noOfClasses)
    # print("////////////////")
    # print(x_train.shape)
    
    return noOfClasses, x_train, x_test, y_train, y_test, x_validation, y_validation, dataGen




































