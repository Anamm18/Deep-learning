# 🌡️ Weather Forecasting with LSTM
### Jena Climate Dataset (2009–2016) · Deep Learning Project

A Streamlit web app that uses an LSTM (Long Short-Term Memory) neural network to forecast weather features — primarily temperature — from the Jena Climate dataset recorded every 10 minutes across 7 years.

---

## 📁 Project Structure

```
weather_forecast(LSTM)/
│
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── jena_climate_2009_2016_csv.zip  # Dataset (place here or update path in app.py)
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Name | Jena Climate Dataset |
| Source | [Kaggle / TensorFlow Datasets](https://www.kaggle.com/datasets/mnassrib/jena-climate) |
| Period | January 2009 – December 2016 |
| Frequency | Every 10 minutes (downsampled to hourly in app) |
| Rows | ~420,550 (raw) · ~70,000 (hourly) |
| Features | 14 (temperature, pressure, humidity, wind speed, etc.) |

### Features in the dataset

| Column | Description |
|---|---|
| `T (degC)` | Temperature in °C *(default prediction target)* |
| `p (mbar)` | Atmospheric pressure |
| `rh (%)` | Relative humidity |
| `wv (m/s)` | Wind velocity |
| `max. wv (m/s)` | Maximum wind velocity |
| `wd (deg)` | Wind direction in degrees |
| `Tpot (K)` | Potential temperature |
| `Tdew (degC)` | Dew point temperature |
| `VPmax/VPact/VPdef (mbar)` | Vapour pressure metrics |
| `sh (g/kg)` | Specific humidity |
| `H2OC (mmol/mol)` | Water vapour concentration |
| `rho (g/m³)` | Air density |

---

## ⚙️ Setup & Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd weather_forecast(LSTM)
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🗂️ Dataset Path

The dataset path is pre-configured in `app.py` at line 41:

```python
DEFAULT_PATH = r"C:\Users\inter\OneDrive\Documents\Deep learning\weather_forecast(LSTM)\jena_climate_2009_2016_csv.zip"
```

If you move the project or run it on a different machine, update this path to wherever your `.zip` file is located. If the file is not found, the app will show a file uploader as a fallback.

---

## 🧠 Model Architecture

The app currently uses **Ridge Regression** as a demo model so no GPU is needed to run it. To use a real LSTM, replace the model block in `app.py` (Section 6) with:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(64, input_shape=(seq_len, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop]
)

y_test_pred = model.predict(X_test)
```

---

## 🖥️ App Features

| Section | Description |
|---|---|
| ⚙️ Sidebar | Configure look-back window, train/val/test split, target feature |
| 1️⃣ Dataset | Auto-loads from hardcoded path, falls back to manual upload |
| 2️⃣ Overview | Row count, feature count, date range, raw data preview |
| 3️⃣ Time Series | Interactive plot of any feature across full dataset |
| 4️⃣ Split View | Color-coded train / validation / test visualization |
| 5️⃣ Preprocessing | MinMaxScaler + sequence creation with shape display |
| 6️⃣ Model | Copy-paste Keras LSTM code + Train button |
| 7️⃣ Metrics | MAE, RMSE, R² on test set |
| 8️⃣ Predictions | Predicted vs Actual line chart with shaded error area |
| 9️⃣ Residuals | Time-series and histogram of prediction errors |
| 🔟 Scatter | Predicted vs Actual scatter with perfect-fit line |
| 📥 Download | Export predictions as `.csv` |

---

## 📐 How the LSTM Pipeline Works

```
Raw CSV (420K rows, 10-min intervals)
        ↓
Downsample → hourly (~70K rows)
        ↓
Chronological split → Train (70%) | Val (15%) | Test (15%)
        ↓
MinMaxScaler → fit on Train only, transform Val & Test
        ↓
Sliding window sequences → shape: (samples, 24 timesteps, 14 features)
        ↓
LSTM model → predict next hour's temperature
        ↓
Inverse transform → back to °C
        ↓
Evaluate: MAE, RMSE, R²
```

> ⚠️ **Key rule:** Never shuffle time-series data. Always split chronologically to prevent data leakage.

---

## 📈 Expected Results (LSTM)

With a properly trained LSTM (2 layers, 64+32 units, 50 epochs):

| Metric | Typical value |
|---|---|
| MAE | ~0.3 – 0.6 °C |
| RMSE | ~0.5 – 0.9 °C |
| R² | ~0.97 – 0.99 |

---

## 🛠️ Requirements

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
```

Python version: **3.9 – 3.11** recommended

---

## 👩‍💻 Author

**Anam Mulla**
B.Tech — Artificial Intelligence & Data Science


---

## 📄 License

This project is for academic and educational purposes.
Dataset credit: Max Planck Institute for Biogeochemistry, Jena, Germany.
