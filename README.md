# Energy Demand Forecaster
XGBoost ML Project | Hourly Energy Consumption | Portfolio Edition

📘 Overview
This project performs Temporal Patterns & Demand Forecasting Analysis on user‑provided hourly electricity datasets.
It uses XGBoost to model and forecast demand, displaying results through an interactive dashboard for easy interpretation of monthly, seasonal, and diurnal patterns.
Additional modules allow dataset comparison and industry intelligence insights to improve grid performance and energy planning.

🧠 Machine Learning & Data Stack
XGBoost — gradient boosting model for demand forecasting
scikit‑learn — evaluation metrics (MSE, MAE, MAPE, SMAPE)
pandas — data manipulation and feature engineering
numpy — numerical computations
holidays — US public holiday detection for temporal features

🖥️ Backend Stack
FastAPI — REST API framework
Uvicorn — ASGI server
python‑multipart — CSV file upload handling

💻 Frontend Stack
React 18 — UI framework
Vite — build tool and dev server
Plotly.js / react‑plotly.js — interactive charts
Axios — HTTP requests to backend
Tailwind CSS — utility‑first styling

🔍 Features Built
Engineered Features — 17 temporal, lag, and rolling statistical features
XGBoost Regressor — trained with temporal train/test split
Evaluation Metrics — RMSE, MAE, MAPE, SMAPE
Interactive Charts — 6 Plotly visualizations for demand trends
CSV Upload — automatic column detection and validation
Company Comparison — compare two datasets for performance analysis
Research Report — temporal pattern insights and forecasting summary

Industry Intelligence — recommendations for grid optimization and technology adoption
📊 Example Insights
Peak Demand: 25,695 MW (Oct 20 2008 14:00)
Dominant Season: Winter (avg 16,964.9 MW)
Model Performance: MAPE 0.93 %, RMSE 186 MW
Feature Importance: lag_1, lag_7, rolling_mean_3, hour, is_weekend

🚀 How It Works
Upload your hourly dataset (.csv)
Train the XGBoost model via the dashboard
View Actual vs Predicted plots and feature importances
Compare datasets across companies
Explore Industry Intelligence for actionable insights

📈 Future Enhancements
Integration with LSTM + XGBoost ensemble for hybrid forecasting
Real‑time data ingestion via API streaming
Expanded regional datasets for comparative analytics
Enhanced grid‑level optimization recommendations
💻 Demo:http://localhost:3000/
