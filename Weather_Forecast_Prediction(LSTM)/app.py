import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import zipfile
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Forecasting — Jena Climate",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Weather Forecasting with LSTM")
st.caption("Jena Climate Dataset (2009–2016) · Predict temperature using sequential deep learning")

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Settings")
    seq_len     = st.slider("Look-back window (hours)", 12, 72, 24, step=6)
    train_ratio = st.slider("Train split (%)", 60, 80, 70, step=5)
    val_ratio   = st.slider("Validation split (%)", 5, 20, 15, step=5)
    downsample  = st.checkbox("Downsample to hourly", value=True)
    target_col_label = st.selectbox(
        "Target feature to predict",
        ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"],
        index=0
    )
    st.divider()
    st.markdown("**Test split:** automatically the remainder")
    test_pct = 100 - train_ratio - val_ratio
    st.info(f"Test: {test_pct}%")

# ── Dataset path ──────────────────────────────────────────────────────────────
DEFAULT_PATH = r"C:\Users\inter\OneDrive\Documents\Deep learning\weather_forecast(LSTM)\jena_climate_2009_2016_csv.zip"

@st.cache_data(show_spinner="Loading dataset...")
def load_data(file_bytes, filename, do_downsample):
    if filename.endswith(".zip"):
        import io
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        import io
        df = pd.read_csv(io.BytesIO(file_bytes))

    df.rename(columns={"Date Time": "Date_Time"}, inplace=True)
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date_Time")

    # Fix known bad wind-speed values
    if "wv (m/s)" in df.columns:
        df["wv (m/s)"] = df["wv (m/s)"].clip(lower=0)
    if "max. wv (m/s)" in df.columns:
        df["max. wv (m/s)"] = df["max. wv (m/s)"].clip(lower=0)

    if do_downsample:
        df = df[::6]   # 10-min → hourly
    return df


def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)


def inv_transform_target(arr_scaled, scaler, n_features, target_idx):
    dummy = np.zeros((len(arr_scaled), n_features))
    dummy[:, target_idx] = arr_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


# ── Dataset loading: auto-load from path or manual upload ─────────────────────
st.subheader("1️⃣ Dataset")

file_bytes = None
source_name = None

if os.path.exists(DEFAULT_PATH):
    st.success(f"✅ Dataset auto-loaded from: `{DEFAULT_PATH}`")
    with open(DEFAULT_PATH, "rb") as f:
        file_bytes = f.read()
    source_name = DEFAULT_PATH
else:
    st.warning("⚠️ Default path not found. Please upload manually.")
    st.caption(f"Expected: `{DEFAULT_PATH}`")
    uploaded = st.file_uploader(
        "Upload `jena_climate_2009_2016.csv` or its `.zip` manually",
        type=["csv", "zip"]
    )
    if uploaded is not None:
        file_bytes = uploaded.read()
        source_name = uploaded.name

if file_bytes is None:
    st.info("Dataset not found. Please upload the file above.")
    st.stop()

df = load_data(file_bytes, source_name, downsample)

# ── Dataset overview ──────────────────────────────────────────────────────────
st.subheader("2️⃣ Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total rows", f"{len(df):,}")
col2.metric("Features", len(df.columns))
col3.metric("Start date", str(df.index[0].date()))
col4.metric("End date",   str(df.index[-1].date()))

with st.expander("Preview raw data"):
    st.dataframe(df.head(50), use_container_width=True)

# ── Time series plot ──────────────────────────────────────────────────────────
st.subheader("3️⃣ Feature Time Series")
plot_col = st.selectbox("Select feature to plot", df.columns.tolist(), index=df.columns.tolist().index(target_col_label))

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df.index, df[plot_col], linewidth=0.6, color="#378ADD", alpha=0.85)
ax.set_title(f"{plot_col} over time", fontsize=12)
ax.set_ylabel(plot_col, fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.2)
fig.tight_layout()
st.pyplot(fig)
plt.close()

# ── Train / Val / Test split ───────────────────────────────────────────────────
st.subheader("4️⃣ Train / Val / Test Split")

features    = df.columns.tolist()
target_idx  = features.index(target_col_label)
n           = len(df)
train_end   = int(n * train_ratio / 100)
val_end     = int(n * (train_ratio + val_ratio) / 100)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

c1, c2, c3 = st.columns(3)
c1.metric("Train rows", f"{len(train_df):,}", f"{train_ratio}% · {train_df.index[0].date()} → {train_df.index[-1].date()}")
c2.metric("Val rows",   f"{len(val_df):,}",   f"{val_ratio}% · {val_df.index[0].date()} → {val_df.index[-1].date()}")
c3.metric("Test rows",  f"{len(test_df):,}",  f"{test_pct}% · {test_df.index[0].date()} → {test_df.index[-1].date()}")

# Split visualization
fig2, ax2 = plt.subplots(figsize=(12, 2.5))
ax2.plot(train_df.index, train_df[target_col_label], color="#378ADD", linewidth=0.5, label="Train")
ax2.plot(val_df.index,   val_df[target_col_label],   color="#1D9E75", linewidth=0.5, label="Validation")
ax2.plot(test_df.index,  test_df[target_col_label],  color="#D85A30", linewidth=0.5, label="Test")
ax2.set_title(f"{target_col_label} — chronological split", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
fig2.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Scaling + sequences ───────────────────────────────────────────────────────
st.subheader("5️⃣ Preprocessing — Scale & Sequence")

with st.spinner("Scaling and creating sequences..."):
    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    val_scaled   = scaler.transform(val_df[features])
    test_scaled  = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(train_scaled, seq_len, target_idx)
    X_val,   y_val   = create_sequences(val_scaled,   seq_len, target_idx)
    X_test,  y_test  = create_sequences(test_scaled,  seq_len, target_idx)

s1, s2, s3 = st.columns(3)
s1.metric("X_train shape", f"{X_train.shape}")
s2.metric("X_val shape",   f"{X_val.shape}")
s3.metric("X_test shape",  f"{X_test.shape}")

st.code(f"""
# Shapes ready for LSTM
X_train: {X_train.shape}   # (samples, timesteps={seq_len}, features={len(features)})
X_val:   {X_val.shape}
X_test:  {X_test.shape}
""", language="python")

# ── Model training ─────────────────────────────────────────────────────────────
st.subheader("6️⃣ Train LSTM Model")

with st.expander("📋 Model architecture (copy-paste ready)"):
    st.code(f"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(64, input_shape=({seq_len}, {len(features)}), return_sequences=True),
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
""", language="python")

train_btn = st.button("🚀 Train Model (uses LinearRegression as demo)", type="primary")

if train_btn or "model_trained" in st.session_state:
    with st.spinner("Training... (demo model — swap in Keras LSTM for real results)"):
        from sklearn.linear_model import Ridge

        X_tr_flat = X_train.reshape(X_train.shape[0], -1)
        X_vl_flat = X_val.reshape(X_val.shape[0], -1)
        X_te_flat = X_test.reshape(X_test.shape[0], -1)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_flat, y_train)

        y_val_pred  = model.predict(X_vl_flat)
        y_test_pred = model.predict(X_te_flat)

        st.session_state["model_trained"] = True
        st.session_state["y_test"]        = y_test
        st.session_state["y_test_pred"]   = y_test_pred
        st.session_state["y_val"]         = y_val
        st.session_state["y_val_pred"]    = y_val_pred
        st.session_state["scaler"]        = scaler
        st.session_state["test_df_idx"]   = test_df.index[seq_len:]

    st.success("✅ Model trained!")

    # ── Evaluation metrics ────────────────────────────────────────────────────
    st.subheader("7️⃣ Evaluation Metrics")

    actual_c    = inv_transform_target(y_test,      scaler, len(features), target_idx)
    predicted_c = inv_transform_target(y_test_pred, scaler, len(features), target_idx)

    mae   = mean_absolute_error(actual_c, predicted_c)
    rmse  = np.sqrt(mean_squared_error(actual_c, predicted_c))
    ss_res = np.sum((actual_c - predicted_c) ** 2)
    ss_tot = np.sum((actual_c - actual_c.mean()) ** 2)
    r2    = 1 - ss_res / ss_tot

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae:.3f} °C")
    m2.metric("RMSE", f"{rmse:.3f} °C")
    m3.metric("R²",   f"{r2:.4f}")

    # ── Prediction vs Actual ──────────────────────────────────────────────────
    st.subheader("8️⃣ Prediction vs Actual")

    n_show = st.slider("Number of test points to display", 50, min(500, len(actual_c)), 200, step=50)
    idx    = st.session_state["test_df_idx"][:n_show]
    act    = actual_c[:n_show]
    pred   = predicted_c[:n_show]

    fig3, ax3 = plt.subplots(figsize=(13, 4))
    ax3.plot(idx, act,  color="#378ADD", linewidth=1.2, label="Actual",    zorder=2)
    ax3.plot(idx, pred, color="#D85A30", linewidth=1.2, label="Predicted",
             linestyle="--", alpha=0.85, zorder=3)
    ax3.fill_between(idx, act, pred, alpha=0.10, color="#D85A30")
    ax3.set_title(f"Predicted vs Actual — {target_col_label} (test set)", fontsize=12)
    ax3.set_ylabel(target_col_label, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.subheader("9️⃣ Residuals (Actual − Predicted)")

    residuals = act - pred
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 3.5))

    ax4a.plot(idx, residuals, color="#888780", linewidth=0.8)
    ax4a.axhline(0, color="#D85A30", linewidth=1, linestyle="--")
    ax4a.set_title("Residuals over time", fontsize=11)
    ax4a.set_ylabel("Error (°C)", fontsize=10)
    ax4a.grid(True, alpha=0.2)

    ax4b.hist(residuals, bins=40, color="#378ADD", alpha=0.75, edgecolor="white")
    ax4b.axvline(0, color="#D85A30", linewidth=1.5, linestyle="--")
    ax4b.set_title("Residuals distribution", fontsize=11)
    ax4b.set_xlabel("Error (°C)", fontsize=10)
    ax4b.grid(True, alpha=0.2)

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.subheader("🔟 Scatter — Predicted vs Actual")

    fig5, ax5 = plt.subplots(figsize=(5, 5))
    ax5.scatter(act, pred, alpha=0.3, s=10, color="#378ADD")
    lo, hi = min(act.min(), pred.min()), max(act.max(), pred.max())
    ax5.plot([lo, hi], [lo, hi], color="#D85A30", linewidth=1.5, linestyle="--", label="Perfect fit")
    ax5.set_xlabel(f"Actual {target_col_label}", fontsize=10)
    ax5.set_ylabel(f"Predicted {target_col_label}", fontsize=10)
    ax5.set_title("Scatter plot", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close()

    # ── Download predictions ──────────────────────────────────────────────────
    st.subheader("📥 Download Predictions")
    results_df = pd.DataFrame({
        "datetime":  st.session_state["test_df_idx"],
        "actual":    actual_c.round(3),
        "predicted": predicted_c.round(3),
        "error":     (actual_c - predicted_c).round(3)
    })
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("Configure settings in the sidebar then click **Train Model** above.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Jena Climate Dataset · LSTM Weather Forecasting Project")