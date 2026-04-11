# 📈 Stock Direction Prediction using Machine Learning

This project uses **historical stock data + technical indicators** to predict the **next-day price direction** (UP/DOWN) of a stock using a **Random Forest Classifier**.

---

## 📊 Project Overview

- Stock: **Larsen & Toubro (LT.NS)**
- Market Index: **NIFTY 50 (^NSEI)**
- Model: **Random Forest Classifier**
- Goal: Predict whether the stock price will go **UP (1)** or **DOWN (0)** the next day

---

## 📉 Visualization

The model output and feature relationships are visualized below:

![Model Plot](50_per.png)

### 🧠 Interpretation

- The dense cluster in the center shows **high overlap of features**
- This indicates:
  - Market behavior is **noisy**
  - Features are **not strongly separable**
- Extreme lines represent **volatile movements**

👉 This explains why stock prediction is inherently difficult.

---

## ⚙️ How It Works

### 1️⃣ Data Collection

- Uses `yfinance` to download:
  - Stock data (LT.NS)
  - NIFTY index data

---

### 2️⃣ Feature Engineering

The model uses the following features:

#### 📌 Market-Based Features
- Previous day return (`y_return`)
- Previous RSI (`y_RSI`)
- Previous NIFTY return (`y_nd_change`)
- Previous Volume (`y_volume`)

#### 📌 Trend Indicators
- 50-day Moving Average (`MA50`)
- 200-day Moving Average (`MA200`)
- Trend (MA50 > MA200)

#### 📌 Target Variable
- `direction`:
  - `1` → Price goes UP
  - `0` → Price goes DOWN

---

### 3️⃣ Technical Indicator (RSI)

- RSI is computed using 14-day rolling averages:
  - Gains & losses
  - Relative Strength (RS)

---

### 4️⃣ Model Training

- Train-test split: **80% train, 20% test**
- No shuffling (important for time-series)

#### Model Used:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
