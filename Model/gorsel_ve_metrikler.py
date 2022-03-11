import numpy as np
from sklearn.metrics import confusion_matrix # değerlendirme metriğinden karışıklık matrisini import ettik
import seaborn as sns #görselleştirmek için 
import matplotlib.pyplot as plt #görselleştirmek için 
from sklearn import metrics






    
def gorsel_ve_metrik(hist, model, x_test, y_test, x_validation,y_validation):
    hist.history.keys()
    
    plt.figure()
    plt.plot(hist.history["loss"], label = "Eğitim Loss") # model çıktılarının içindeki histogram (hist)
    plt.plot(hist.history["val_loss"], label = "Val Loss") # kayıp doğrulama 
    plt.legend()
    plt.show()
    
    
    plt.figure()
    plt.plot(hist.history["accuracy"], label = "Eğitim accuracy") # doğruluk
    plt.plot(hist.history["val_accuracy"], label = "Val accuracy")  # doğruluk doğrulama
    plt.legend()
    plt.show()
    
    
    
    score = model.evaluate(x_test, y_test, verbose = 1) # kayıp ve doğruluk için ayarla
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    
    
    
    x_test_pred = model.predict(x_test)
    x_test_pred_pred_class = np.argmax(x_test_pred, axis = 1)
    y_test_true = np.argmax(y_test, axis = 1)
    cm0 = confusion_matrix(y_test_true, x_test_pred_pred_class)
    f, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm0, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("cm test")
    plt.show()
    
    print("-----------------------Test-------------------")
    print(metrics.classification_report(y_test_true, x_test_pred_pred_class, digits=3))
    
    
    # confusion matrix
    
    y_pred = model.predict(x_validation) # Bir sınıflandırıcı tarafından döndürülen tahmini hedefler.
    y_pred_class = np.argmax(y_pred, axis = 1)
    Y_true = np.argmax(y_validation, axis = 1) # Kesin referans (doğru) hedef değerler.
    cm = confusion_matrix(Y_true, y_pred_class)
    f, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("cm validation")
    plt.show()
    
    # labels = ["Babesia", "Leishmania", "Plasmodium", "Toxoplasma_400X", "Toxoplasma_1000X", "Trichomonad", "Trypanosome"]
    
    # Print the precision and recall, among other metrics
    print("------------------------Validation------------")
    print(metrics.classification_report(Y_true, y_pred_class, digits=3)) # doğru ve tahmin edilenler fonksiyona giriyor. digits hassasiyeti ayarlar
    
    
    # model.save('parazit_model_11.h5')