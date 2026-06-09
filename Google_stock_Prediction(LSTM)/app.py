import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Google Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #13161e;
        border-right: 1px solid #1e2330;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f1624 0%, #1a2540 100%);
        border: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f4ff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #6b7fa3;
        font-size: 0.95rem;
        margin: 0;
    }
    .accent {
        color: #4f8ef7;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: #13161e;
        border: 1px solid #1e2330;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6b7fa3;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f4ff;
    }
    .metric-delta-pos { color: #34d399; font-size: 0.8rem; }
    .metric-delta-neg { color: #f87171; font-size: 0.8rem; }

    /* Section headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #8ba3c7;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e2330;
    }

    /* Info box */
    .info-box {
        background: #0f1e35;
        border-left: 3px solid #4f8ef7;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #8db0e0;
        margin: 0.5rem 0 1rem 0;
    }

    /* Table styling */
    .dataframe {
        background: #13161e !important;
        color: #c8d5e8 !important;
        border-radius: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #1a3a6b;
        color: #c8dcf8;
        border: 1px solid #2a5aab;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #1e4a8a;
        border-color: #4f8ef7;
        color: #ffffff;
    }

    /* Uploader */
    .stFileUploader {
        background: #13161e;
        border-radius: 10px;
        border: 1px dashed #2a3450;
    }

    /* Chart background transparency */
    .element-container iframe { background: transparent; }

    /* Selectbox / slider labels */
    label { color: #8ba3c7 !important; font-size: 0.88rem !important; }

    /* Spinner text */
    .stSpinner > div > div { border-top-color: #4f8ef7 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────
TIMESTEPS = 60

@st.cache_resource
def load_keras_model(path):
    return load_model(path)

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def build_sequences(scaled, timesteps=60):
    X, y = [], []
    for i in range(timesteps, len(scaled)):
        X.append(scaled[i - timesteps:i, 0])
        y.append(scaled[i, 0])
    return np.array(X).reshape(-1, timesteps, 1), np.array(y)

def styled_chart(dates_train, train_vals,
                 dates_test,  real_vals, pred_vals,
                 show_train=True):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#0d0f14')

    if show_train:
        ax.plot(dates_train, train_vals,
                color='#2a3e60', linewidth=1, label='Training data', alpha=0.7)

    ax.plot(dates_test, real_vals,
            color='#4f8ef7', linewidth=2, label='Actual price')
    ax.plot(dates_test, pred_vals,
            color='#34d399', linewidth=2, linestyle='--', label='Predicted price')

    # Fill between actual and predicted
    ax.fill_between(dates_test, real_vals, pred_vals,
                    alpha=0.08, color='#34d399')

    # Vertical divider at train/test boundary
    if show_train and len(dates_train) > 0:
        ax.axvline(x=dates_train.iloc[-1], color='#3a4a6a',
                   linestyle=':', linewidth=1, alpha=0.8)
        ax.text(dates_train.iloc[-1], ax.get_ylim()[1],
                ' train / test', color='#4a6a9a',
                fontsize=8, va='top')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, color='#6b7fa3', fontsize=9)
    plt.yticks(color='#6b7fa3', fontsize=9)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.grid(color='#1e2330', linewidth=0.6, linestyle='-')
    ax.tick_params(colors='#6b7fa3', length=0)

    legend = ax.legend(facecolor='#13161e', edgecolor='#1e2330',
                       labelcolor='#c8d5e8', fontsize=9, loc='upper left')
    ax.set_ylabel('Stock Price (USD)', color='#6b7fa3', fontsize=9)

    plt.tight_layout()
    return fig


# ─── File paths — all files must be in the same folder as app.py ───────────────

BASE_DIR   = r"C:\Users\inter\OneDrive\Documents\Deep learning\Google_stock_prediction(LSTM)"
MODEL_PATH = BASE_DIR + r"\stock_prediction_model.keras"
TRAIN_PATH = BASE_DIR + r"\Google_Stock_Price_Train.csv"
TEST_PATH  = BASE_DIR + r"\Google_Stock_Price_Test.csv"
# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**Display options**")
    show_train  = st.toggle("Show training curve",  value=True)
    show_table  = st.toggle("Show predictions table", value=True)
    timesteps   = st.slider("Sequence length (timesteps)", 30, 120, 60, 10)
    st.markdown("---")
    st.markdown(
        "<div style='color:#3a4a6a;font-size:0.78rem'>"
        "LSTM · 4 layers · 50 units<br>Dropout 0.2 · Adam · MSE<br>"
        "Trained Jan 2012 – Dec 2016"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">📈 Google Stock <span class="accent">Price Predictor</span></p>
  <p class="hero-sub">LSTM · RNN · Keras &nbsp;|&nbsp; 4-layer stacked LSTM trained on 2012–2016 data</p>
</div>
""", unsafe_allow_html=True)


# ─── Load assets from local paths ──────────────────────────────────────────────
try:
    model_src = MODEL_PATH
    train_src = TRAIN_PATH
    test_src  = TEST_PATH

    with st.spinner("Loading model…"):
        model = load_keras_model(model_src)

    train_df = load_csv(train_src)
    test_df  = load_csv(test_src)

    # ── Preprocessing ────────────────────────────────────────────────────────
    train_prices = train_df[['Open']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices)

    # Build test input (concat last `timesteps` train rows)
    real_prices = test_df[['Open']].values
    combined = pd.concat(
        (train_df['Open'], test_df['Open']), axis=0
    )
    n_test   = len(test_df)
    inputs   = combined.values[-(n_test + timesteps):].reshape(-1, 1)
    inputs   = scaler.transform(inputs)

    X_test = []
    for i in range(timesteps, timesteps + n_test):
        X_test.append(inputs[i - timesteps:i, 0])
    X_test = np.array(X_test).reshape(-1, timesteps, 1)

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Running prediction…"):
        pred_scaled = model.predict(X_test, verbose=0)
    predicted = scaler.inverse_transform(pred_scaled)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = float(np.mean(np.abs(real_prices - predicted)))
    rmse = float(np.sqrt(np.mean((real_prices - predicted) ** 2)))
    mape = float(np.mean(np.abs((real_prices - predicted) / real_prices)) * 100)
    last_real = float(real_prices[-1, 0])
    last_pred = float(predicted[-1, 0])
    delta_pct = (last_pred - last_real) / last_real * 100

    # ── Metric cards ─────────────────────────────────────────────────────────
    delta_cls = "metric-delta-pos" if delta_pct >= 0 else "metric-delta-neg"
    delta_sym = "▲" if delta_pct >= 0 else "▼"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Last actual price</div>
        <div class="metric-value">${last_real:,.2f}</div>
        <div class="{delta_cls}">{delta_sym} {abs(delta_pct):.2f}% vs prediction</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Last predicted price</div>
        <div class="metric-value">${last_pred:,.2f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">${mae:.2f}</div>
        <div style="color:#6b7fa3;font-size:0.78rem">Mean absolute error</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">${rmse:.2f}</div>
        <div style="color:#6b7fa3;font-size:0.78rem">Root mean squared error</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">MAPE</div>
        <div class="metric-value">{mape:.2f}%</div>
        <div style="color:#6b7fa3;font-size:0.78rem">Mean abs % error</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Price chart</p>', unsafe_allow_html=True)

    fig = styled_chart(
        train_df['Date'], train_df['Open'].values,
        test_df['Date'],  real_prices.flatten(), predicted.flatten(),
        show_train=show_train
    )
    st.pyplot(fig, use_container_width=True)

    # ── Predictions table ─────────────────────────────────────────────────────
    if show_table:
        st.markdown('<p class="section-header">Daily predictions</p>',
                    unsafe_allow_html=True)

        results_df = pd.DataFrame({
            'Date':           test_df['Date'].dt.strftime('%d %b %Y'),
            'Actual ($)':     real_prices.flatten().round(2),
            'Predicted ($)':  predicted.flatten().round(2),
            'Error ($)':      (real_prices - predicted).flatten().round(2),
            'Error (%)':      ((real_prices - predicted) / real_prices * 100).flatten().round(2),
        })

        def color_error(val):
            color = '#34d399' if abs(val) < 5 else '#f87171'
            return f'color: {color}'

        styled = (
    results_df.style
    .map(color_error, subset=['Error (%)'])
    .set_properties(**{
        'background-color': '#13161e',
        'color': '#c8d5e8',
        'border-color': '#1e2330',
    })
    .format({
        'Actual ($)': '${:.2f}',
        'Predicted ($)': '${:.2f}',
        'Error ($)': '{:+.2f}',
        'Error (%)': '{:+.2f}%'
    })

        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download button
        csv_out = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download predictions CSV",
            data=csv_out,
            file_name="google_stock_predictions.csv",
            mime="text/csv"
        )

    # ── Model summary ─────────────────────────────────────────────────────────
    with st.expander("🔍 Model architecture"):
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines), language="text")

    # ── About ─────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Architecture** — 4 stacked LSTM layers (50 units each) with 0.2 Dropout
        after every layer, followed by a single Dense output neuron.

        **Preprocessing** — The `Open` price column is scaled to [0, 1] using
        MinMaxScaler fitted on training data only. Sliding windows of 60 trading
        days are used as input sequences.

        **Training data** — Jan 2012 – Dec 2016 (1257 rows).
        **Test data** — Jan 2017 (20 rows).

        **Metrics**
        - **MAE** — average dollar error per day
        - **RMSE** — penalises large errors more heavily
        - **MAPE** — percentage error, scale-independent
        """)

except FileNotFoundError as e:
    st.error(f"Default file not found: {e}. Please upload files in the sidebar.")
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)