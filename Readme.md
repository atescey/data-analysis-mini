# Higgs Boson Classification

Bu proje, Higgs Boson veri setiyle basit bir ikili sınıflandırma modeli geliştirmek için hazırlanmıştır. 
CERN’deki ATLAS deneyinde, parçacık çarpışmalarından gelen büyük veriyi ayırmak için benzer makine öğrenmesi 
yöntemleri kullanılmaktadır. Bu çalışma, o sürecin sadeleştirilmiş bir örneğidir.

---

## Amaç

- Veri setini yüklemek ve incelemek  
- Eksik değerleri temizlemek ve ölçeklendirmek  
- Random Forest ile “signal” ve “background” sınıflarını tahmin etmek  
- ROC Curve ve Feature Importance grafikleri oluşturmak  

---

## Kullanılan Teknolojiler

- Python  
- Pandas  
- NumPy  
- scikit-learn  
- Matplotlib  

---

## Yapılanlar

- Veri setinin indirilmesi ve okunması  
- Etiket dağılımının kontrol edilmesi  
- Eksik değer işlemesi  
- Model eğitimi  
- Accuracy, Precision, Recall, F1-score hesaplamaları  
- ROC Curve ve Feature Importance grafiklerinin üretilmesi  

---

## Görseller

### Etiket Dağılımı  
![Etiket Dağılımı](./plots/plot1.png)

### ROC Curve  
![ROC Curve](./plots/roc_curve.png)
