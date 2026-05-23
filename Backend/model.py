import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import holidays
import joblib
import os

FEATURE_COLS = [
    "hour", "dayofweek", "month", "year", "dayofyear", "weekofyear",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_3", "rolling_mean_7", "rolling_mean_30",
    "rolling_std_3", "rolling_std_7", "rolling_std_30",
]

MODEL_PATH = "model.joblib"


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    us_hols = holidays.US(years=df["Datetime"].dt.year.unique().tolist())
    df["is_holiday"] = df["Datetime"].dt.date.astype(str).map(
        lambda d: 1 if pd.Timestamp(d) in us_hols else 0
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Datetime").reset_index(drop=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Fill missing hourly timestamps
    full_range = pd.date_range(df["Datetime"].min(), df["Datetime"].max(), freq="h")
    df = df.set_index("Datetime").reindex(full_range).rename_axis("Datetime").reset_index()
    df["MW"] = df["MW"].ffill()

    df["hour"]       = df["Datetime"].dt.hour
    df["dayofweek"]  = df["Datetime"].dt.dayofweek
    df["month"]      = df["Datetime"].dt.month
    df["year"]       = df["Datetime"].dt.year
    df["dayofyear"]  = df["Datetime"].dt.dayofyear
    df["weekofyear"] = df["Datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df = add_holidays(df)

    df["lag_1"]  = df["MW"].shift(1)
    df["lag_7"]  = df["MW"].shift(7 * 24)
    df["lag_30"] = df["MW"].shift(30 * 24)

    for w in [3, 7, 30]:
        hrs = w * 24
        df[f"rolling_mean_{w}"] = df["MW"].shift(1).rolling(hrs, min_periods=1).mean()
        df[f"rolling_std_{w}"]  = df["MW"].shift(1).rolling(hrs, min_periods=1).std()

    return df.dropna()


def train(df: pd.DataFrame, params: dict = None):
    df_feat = engineer_features(df)
    train_df = df_feat[df_feat["year"] < 2014]
    test_df  = df_feat[df_feat["year"] == 2014]

    if len(test_df) == 0:
        split = int(len(df_feat) * 0.9)
        train_df, test_df = df_feat.iloc[:split], df_feat.iloc[split:]

    X_tr, y_tr = train_df[FEATURE_COLS], train_df["MW"]
    X_te, y_te = test_df[FEATURE_COLS],  test_df["MW"]

    default_params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, tree_method="hist",
    )
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    preds = model.predict(X_te)
    metrics = {
        "rmse":  float(np.sqrt(mean_squared_error(y_te, preds))),
        "mae":   float(mean_absolute_error(y_te, preds)),
        "mape":  float(mape(y_te.values, preds)),
        "smape": float(smape(y_te.values, preds)),
    }

    importance = [
        {"feature": f, "importance": float(v)}
        for f, v in sorted(
            zip(FEATURE_COLS, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]

    test_result = test_df[["Datetime", "MW"]].copy()
    test_result["predicted"] = preds
    test_result["Datetime"] = test_result["Datetime"].astype(str)

    joblib.dump({"model": model, "df_feat": df_feat}, MODEL_PATH)

    return metrics, importance, test_result.to_dict(orient="records"), df_feat


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2001-01-01", "2014-12-31 23:00:00", freq="h")
    n = len(dates)
    hour  = dates.hour
    month = dates.month
    dow   = dates.dayofweek
    base     = 30000
    hourly   = 5000  * np.sin((hour - 6) * np.pi / 12)
    seasonal = 8000  * np.cos((month - 7) * np.pi / 6)
    weekly   = -2000 * (dow >= 5).astype(float)
    trend    = np.linspace(0, 5000, n)
    noise    = np.random.normal(0, 800, n)
    mw = np.clip(base + hourly + seasonal + weekly + trend + noise, 10000, 65000)
    return pd.DataFrame({"Datetime": dates, "MW": mw})
