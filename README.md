# LSTM Stock Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**Prediksi harga saham BBRI menggunakan arsitektur LSTM (Long Short-Term Memory) berbasis PyTorch dengan antarmuka Streamlit.**

[Demo](#demo) · [Instalasi](#instalasi) · [Cara Pakai](#cara-pakai) · [Arsitektur](#arsitektur)

</div>

---

## Fitur Utama

- **Prediksi harga penutupan** saham hari berikutnya
- **Visualisasi** perbandingan harga aktual vs prediksi
- **Sistem autentikasi** user (login/register via SQLite)
- **Riwayat prediksi** tersimpan per akun
- **Dua mode data**: dataset penelitian statis (1000 hari) & live via yFinance
- **Save/load model** artifacts (state dict, scaler, config, metrics)

---

## Struktur Proyek

```
lstm-stock-prediction/
│
├── app.py                  # Aplikasi utama Streamlit
├── model_logic.py          # Pipeline LSTM: data → train → predict
├── database_helper.py      # SQLite: autentikasi & riwayat prediksi
│
├── data/
│   └── dataset_bbri_1000_rounded.csv   # Dataset penelitian BBRI
│
├── models/
│   ├── model_state.pt      # Model weights (PyTorch)
│   ├── scaler.joblib       # MinMaxScaler tersimpan
│   ├── config.json         # Hyperparameter config
│   └── metrics.json        # MSE & RMSE hasil training
│
├── notebooks/
│   └── explorasi_lstm.ipynb  # Notebook eksplorasi & eksperimen
│
├── utils/
│   └── __init__.py
│
├── requirements.txt
└── .gitignore
```

---

## Instalasi

```bash
# Clone repo
git clone https://github.com/MuhammadSamman/lstm-stock-prediction.git
cd lstm-stock-prediction

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

---

## Requirements

```
torch>=2.0.0
streamlit>=1.28.0
yfinance>=0.2.18
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

---

## Arsitektur

```
Input (OHLCV × 60 hari)
        ↓
  MinMaxScaler
        ↓
 LSTM Layer(s)
 [hidden=64, layers=1]
        ↓
  Linear Head
        ↓
 Prediksi Close (t+1)
```

| Parameter     | Nilai Default |
|---------------|--------------|
| Window        | 60 hari      |
| Epochs        | 50           |
| Learning Rate | 0.001        |
| Hidden Size   | 64           |
| Batch Size    | 32           |
| Train/Test    | 80% / 20%    |

---

## Evaluasi

| Metrik     | Keterangan                          |
|-----------|-------------------------------------|
| MSE       | Mean Squared Error pada data test   |
| RMSE      | Root MSE — dalam satuan harga (IDR) |

---

## Akun Default

| Username | Password | Role  |
|----------|----------|-------|
| admin    | admin    | admin |

> Registrasi akun baru hanya dapat dilakukan oleh admin.

---

## Lisensi

MIT License — lihat [LICENSE](LICENSE)

---

<div align="center">
Made with by <a href="https://github.com/MuhammadSamman">Muhammad Samman</a>
</div>
