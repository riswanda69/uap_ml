# Klasifikasi Rumah Adat dengan ResNet50 dan CNN

## Overview Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi gambar otomatis yang mampu mengenali dan mengklasifikasikan gambar rumah adat ke dalam lima kategori yang berbeda. Sistem ini dirancang untuk mendukung pelestarian budaya dengan memanfaatkan teknologi deep learning untuk identifikasi rumah adat dari berbagai wilayah di Indonesia.

**Link Dataset**: Dataset Rumah Adat. https://www.kaggle.com/datasets/rariffirmansah/rumah-adat

**Preprocessing**: Dataset diproses menggunakan resizing, normalization, dan augmentation untuk meningkatkan performa model.

**Model yang Digunakan**: 
1. ResNet50 sebagai model pretrained.
2. CNN khusus dengan 4 convolutional block.

## Arsitektur Model

##ResNet50

ResNet50 digunakan sebagai backbone pretrained model dengan beberapa layer tambahan:

- Global Average Pooling.
- Fully connected layer (512 neurons).
- Dropout untuk regularisasi.
- Softmax output layer untuk 5 kelas.

## CNN Custom
Empat blok convolutional:
- Conv2D, Batch Normalization, MaxPooling.
- Dropout untuk regularisasi.
Fully connected layer (512 neurons).
Output layer dengan softmax activation.

## Overview Dataset

Dataset yang digunakan terdiri atas gambar rumah adat yang dibagi ke dalam lima kelas berbeda. Dataset ini diproses dengan langkah-langkah berikut:

1. Augmentasi:
   Rotasi, perubahan kecerahan, kontras, saturasi, dan flipping horizontal.

2. Pembagian Dataset:
   70% Training, 15% Validation, dan 15% Testing.
   
4. Normalisasi:
   Rescaling ke rentang [0, 1] (1./255) untuk model CNN, dan preprocessing khusus ResNet untuk pretrained model.

## Proses Preprocessing & Training

**Preprocessing**
1. Resizing: Mengubah ukuran gambar menjadi 224x224 px.
2. Augmentasi: Menambahkan variasi pada dataset dengan teknik seperti rotasi, flipping, dan zoom.
3. Splitting: Membagi dataset ke dalam tiga set (Training, Validation, Testing).

## Modelling

## ResNet50

1. Menggunakan arsitektur ResNet50 pretrained dengan ImageNet.
2. Dua tahap pelatihan:
   - Freeze base model: Melatih hanya layer tambahan.
   - Fine-tuning: Membuka 30 layer terakhir untuk pelatihan ulang dengan learning rate rendah.

## CNN Custom
1. Dibangun dari nol dengan empat blok convolutional.
2. Dropout pada beberapa layer untuk regularisasi.
3. Optimizer: Adam dengan mixed precision untuk efisiensi.

## Hasil Evaluasi Model
## CNN Custom
Classification Report
accuracy                               1.00      1800
macro avg          1.00      1.00      1.00      1800
weighted avg       1.00      1.00      1.00      1800

Confusion Matrix menunjukkan distribusi prediksi model untuk masing-masing kelas:
![image](https://github.com/user-attachments/assets/1ba43021-5cd6-4fc4-8970-3913d832a4d0)

Learning Curve
![image](https://github.com/user-attachments/assets/fd79ad7b-a8c0-4b99-9d98-70e256241e89)


## ResNet50
Classification Report
    accuracy                           1.00      2400
   macro avg       1.00      1.00      1.00      2400
weighted avg       1.00      1.00      1.00      2400

Confusion Matrix menunjukkan distribusi prediksi model untuk masing-masing kelas:
![image](https://github.com/user-attachments/assets/98673649-c113-45eb-901b-133d81551e36)


Learning Curve
![image](https://github.com/user-attachments/assets/361154b1-b966-4461-90a8-149790e7fa4e)




## Kesimpulan
1. ResNet50 menunjukkan performa lebih tinggi dibandingkan model CNN custom pada dataset ini.
2. Augmentasi data sangat membantu mengurangi overfitting pada model.

Link Model : https://drive.google.com/drive/folders/143wqSktzoGMU08rkAgTy6ihhywowPEna?usp=sharing
