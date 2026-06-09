# Google Stock Price Prediction — RNN / LSTM

A deep learning project that predicts Google (Alphabet) stock opening prices using an LSTM neural network built with Keras, served through an interactive Streamlit dashboard.

---

## Overview

This project demonstrates an end-to-end time-series forecasting pipeline for financial data. Using five years of historical Google stock prices as training data, an LSTM model learns the temporal patterns in daily opening prices and predicts prices for January 2017. The results are visualised in a Streamlit dashboard that displays the prediction chart, error metrics, and a day-by-day comparison table.

---

## Project Structure

```
Google_stock_prediction(LSTM)/
│
├── app.py                               — Streamlit dashboard
├── Google_stock_prediction_LSTM_.ipynb  — Training notebook
├── stock_prediction_model.keras         — Saved trained model
├── Google_Stock_Price_Train.csv         — Training data (Jan 2012 – Dec 2016)
├── Google_Stock_Price_Test.csv          — Test data (Jan 2017)
└── README.md
```

---

## Dataset

Two CSV files are used in this project. The training file covers January 2012 through December 2016 and contains 1257 rows. The test file covers January 2017 and contains 20 rows. Both files share the same columns — Date, Open, High, Low, Close, and Volume — but only the Open price column is used as the input feature. This is the standard univariate benchmark setup for this dataset.

---

## How It Works

### Data loading and exploration

The training and test CSVs are loaded into pandas DataFrames. The Open price column is extracted as the sole input feature, giving an array of shape (1257, 1) for training.

### Feature scaling

Raw stock prices are compressed into the range [0, 1] using MinMaxScaler. This step is essential because LSTM layers use sigmoid and tanh activation functions internally, which operate most efficiently when inputs fall within this bounded range. Feeding large unbounded values into those activations causes saturation — the gradients become near-zero and the model stops learning effectively.

The scaler is fitted exclusively on training data and then applied to the test data using that same fit. This prevents data leakage, which would occur if the model had any indirect knowledge of the test set's scale during training.

### Sliding window sequences

The model is trained on sequences rather than individual data points. For each position in the training data, a window of the previous 60 days of scaled prices is used as input, and the price on the next day is the target output. This gives the model the temporal context it needs to learn how prices evolve over time. The choice of 60 days corresponds to roughly three calendar months of trading days, which captures both short-term noise and medium-term trends.

### Model architecture

The model is a sequential neural network with a single LSTM layer of 10 units followed by a Dense output layer with 1 unit. The LSTM layer processes the 60-day input sequence and produces a single hidden state that summarises the temporal information. The Dense layer maps that summary to a single predicted price value. The model is compiled with the Adam optimiser and mean squared error as the loss function, then trained for 50 epochs with a batch size of 1.

### Why LSTM over a plain RNN

Plain recurrent networks suffer from the vanishing gradient problem when sequences are long — gradients shrink to near-zero as they are propagated back through many timesteps, so the network cannot learn long-range dependencies. LSTMs solve this by using three gating mechanisms (forget gate, input gate, output gate) that control what information to retain, update, or discard at each step. This allows the model to remember relevant patterns from up to 60 days back without the gradient signal collapsing.

### Test sequence preparation

The test set contains only 20 rows, but the model requires a 60-day input window for every prediction. To generate the first test window, the last 60 rows of training data are prepended to the test data before scaling. This ensures that the model has a complete historical context when making its first prediction for January 2017, and all 20 test predictions follow from there.

### Prediction and inverse transform

The model outputs scaled values in [0, 1]. After prediction, the MinMaxScaler's inverse transform is applied to convert those scaled outputs back into real USD prices, which are then compared against the actual opening prices from the test CSV.

### Inference function

The notebook includes a reusable inference function that accepts any 60 historical Open prices, scales them, feeds them to the model, and returns the predicted next-day price in USD. This makes it straightforward to use the saved model for new predictions without re-running the full training pipeline.

---

## Streamlit Dashboard

The dashboard loads all files automatically from the project folder — no manual uploads are needed. It displays a dark-themed price chart comparing the training history, actual January 2017 prices, and the model's predictions. Five summary metrics are shown at the top: the last actual price, the last predicted price, mean absolute error, root mean squared error, and mean absolute percentage error. Below the chart, a day-by-day table lists every test date with its actual price, predicted price, dollar error, and percentage error. Rows with an error below 5% are highlighted green; those above 5% are highlighted red. A download button exports the full predictions table as a CSV file.

---

## Key Design Choices

**MinMaxScaler over StandardScaler** — StandardScaler centres data around zero with a standard deviation of one, which can produce values as extreme as −3 or +4. Those values push LSTM sigmoid and tanh activations into their flat saturation zones, causing vanishing gradients. MinMaxScaler keeps all values in [0, 1], which aligns precisely with the activation function ranges and avoids this problem. It also makes no assumption about the underlying distribution, which matters because stock prices are not normally distributed — they are right-skewed and non-stationary.

**60-day lookback window** — A window that is too short gives the model too little context to distinguish signal from noise. A window that is too long makes training slow and can introduce irrelevant history. 60 trading days, approximately three calendar months, is the standard choice for this dataset and provides a good balance.

**MSE loss** — Mean squared error penalises large prediction errors more heavily than small ones because of the squaring. For stock price prediction, large misses are more costly than small ones, so this loss function aligns well with the problem.

**Adam optimiser** — Adam adapts the learning rate individually for each parameter based on estimates of first and second moments of the gradients. This makes it well-suited to the noisy, sparse gradient landscape typical of financial time-series data, and it generally converges faster than plain stochastic gradient descent.

---

## Running the Project

Install the required packages and launch the dashboard:

```
pip install tensorflow streamlit scikit-learn matplotlib pandas numpy
streamlit run app.py
```

Before running, update the BASE_DIR path at the top of app.py to point to the folder where your project files are stored on your machine.

---

## Possible Improvements

Adding the High, Low, Close, and Volume columns as additional input features would turn this into a multivariate LSTM, giving the model richer information at each timestep. Stacking multiple LSTM layers with dropout regularisation between them would allow the network to learn more abstract temporal representations and reduce overfitting. Early stopping and learning rate scheduling callbacks would make training more robust. Replacing the single train/test split with walk-forward validation would give a more realistic estimate of how the model performs on unseen data over time. Finally, extending the model to predict multiple days ahead simultaneously, rather than one day at a time, would make it more practically useful.

---

## License

This project is for educational purposes. The dataset is a publicly available Google stock price benchmark widely used in deep learning courses.
