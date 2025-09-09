# **Hotel Booking Cancelation Prediction ML Model**
Analisis data dan prediksi (menggunakan machine learning) mengenai pembatalan reservasi booking hotel menggunakan [dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data). 

![alt text](https://github.com/qtash/Booking-Cancellation-Classification/blob/main/Tableau%20Dashboard.png)

## **Background**
Pembatalan pemesanan merupakan salah satu tantangan terbesar dalam industri perhotelan karena berdampak langsung pada tingkat hunian dan pendapatan. Kasus pembatalan (khususnya yang terjadi mendekati tanggal kedatangan) membuat hotel kesulitan. Kompleksitas masalah semakin tinggi karena alasan pembatalan sangat beragam, mulai dari perubahan rencana perjalanan hingga preferensi pelanggan, sehingga sulit diantisipasi tanpa analisis mendalam. 

Oleh karena itu, pemanfaatan model machine learning untuk memprediksi potensi pembatalan menjadi solusi penting agar hotel dapat merancang strategi harga, kebijakan pembatalan, dan penawaran yang lebih tepat sasaran, sehingga kerugian dapat ditekan sekaligus meningkatkan kepuasan serta loyalitas tamu.

### **Problem**
Tingginya tingkat pembatalan pemesanan kamar hotel menjadi tantangan bagi pihak hotel karena pembatalan yang terjadi mendekati waktu kedatangan sulit untuk diantisipasi. 

### **Goals & Strategy**
Mengidentifikasi variabel-variable yang paling berpengaruh terhadap kemungkinan pembatalan melalui analisis konvensional dan prediksi menggunakan machine learning. 

Hasil analisis dan prediksi ini diharapkan dapat mendukung pengambilan keputusan dalam strategi harga, kebijakan pembatalan, dan alokasi kamar agar lebih efektif dan adaptif terhadap perilaku pelanggan, yang kemudian dapat mengurangi kerugian akibat pembatalan dengan mengidentifikasi pemesanan berisiko tinggi sejak awal.

## **Current Best Model**
Tuned XGBoost model dengan skor F0.5 0.73 dan Precision 0.75. 
#### Fitur/variabel paling berpengaruh: required_car_parking_spaces

## **Conclusion & Recommendation**
Domiansi kepentingan finansial, karakteristik reservasi, pola geografis, dan kebutuhan lahan parkir tamu berpengaruh besar terhadap probabilitas pembatalan reservasi hotel. Risiko pembatalan ini dapat dikelola secara efektif melalui kebijakan deposit disamping mengidentifikasi segmen berisiko pembatalan tinggi lainnya, yaitu saluran pemesanan dan demografi tamu.

Model terbaik yang dihasilkan mampu mendeteksi 65.3% dari seluruh pembatalan yang benar-benar terjadi (berdasarkan recall-nya), dan dari semua prediksi pada training set yang diberikan, 75.6% nya benar-benar akan dibatalkan (berdasarkan precision-nya). Model masih dapat dikembangkan dengan meningkatkan kualitas dataset, mencoba algoritma ML yang dapat meningkatkan recall scoretanpa mengorbankan precision score, serta dengan cara mengimplementasi sistem real-time monitoring untuk terus mengupdate model dengan data yang baru. 

## **Links**
- [Link Tableau](https://public.tableau.com/views/hotel_cancellation/Dashboard?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link)
- Link Streamlit : ((Deploy model melalui GitHub tidak berhasil. Script streamlit terlampir.))
     - [ML models](https://drive.google.com/drive/folders/13m_BuSZVscoqYl9kvIQloVwWktbu2mgl?usp=sharing)

****
- Acknowledgement: Proyek ini diinisasi untuk Final Project pada Bootcamp Data Science & Machine Learning Purwadhika Digital School.
- Created by Team Alpha 
    - Afifa Kinasih Anggitan Mahanani 
    - Muhammad Ilham Pratama 
    - Qonita Shobrina

