# 📈 Amazon Revenue & Profitability Forecasting

Weekly revenue and profit forecasting using SARIMA, Prophet, XGBoost and LSTM on Amazon e-commerce data.

## 🚀 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kaanaltay1/amazon-revenue-forecasting)

👉 **[Click here to open the app](https://huggingface.co/spaces/kaanaltay1/amazon-revenue-forecasting)**

## 📌 Project Overview

This project builds an end-to-end time series forecasting system on real Amazon e-commerce order data. It combines four different forecasting models and provides an interactive Streamlit dashboard for analysis and prediction.

## 🤖 Models Used

| Model | Description |
|-------|-------------|
| **SARIMA** | Seasonal ARIMA for trend and seasonality |
| **Prophet** | Facebook Prophet with holiday effects |
| **XGBoost** | Gradient boosting with lag/rolling features |
| **LSTM** | Deep learning with long-term memory |
| **Ensemble** | Weighted combination of all models |

## 📊 Features

- Upload your own Amazon order CSV files or use built-in demo data
- Forecast revenue and/or profit for up to 26 weeks ahead
- Compare model performances (MAE, RMSE, MAPE)
- Confidence interval bands for uncertainty estimation
- Interactive Plotly charts

## 🗂️ Project Structure

```
amazon-revenue-forecasting/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version (3.11)
└── README.md
```

## ⚙️ Installation & Local Run

```bash
git clone https://github.com/kaanaltay-hub/amazon-revenue-forecasting.git
cd amazon-revenue-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements

- Python 3.11
- streamlit, pandas, numpy, plotly
- scikit-learn, statsmodels
- prophet, xgboost, tensorflow

## 📁 Data Format

The app accepts two CSV files from Amazon Seller Central:
- `amazon_orders_2023_time_series.csv`
- `df_time_series.csv`

Or use the built-in **Demo Data** option from the sidebar.

## 👨‍💻 Author

**Kaan Altay**  
Data Scientist  
[GitHub](https://github.com/kaanaltay-hub) · [Live Demo](https://huggingface.co/spaces/kaanaltay1/amazon-revenue-forecasting)
