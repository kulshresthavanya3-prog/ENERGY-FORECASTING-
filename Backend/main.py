from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import traceback
import holidays as hol_lib
from model import train, generate_demo_data

app = FastAPI(title="PJM Energy Forecaster API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {"trained": False, "metrics": None, "importance": None,
         "predictions": None, "df_feat": None}

compare_state = {"ready": False, "profile_a": None, "profile_b": None}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _parse_csv(content: bytes, filename: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot read {filename}: {e}")

    # find datetime col
    dt_col = next(
        (c for c in raw.columns if c.strip().lower() in ("datetime","date","timestamp","time")),
        raw.columns[0]
    )
    try:
        raw[dt_col] = pd.to_datetime(raw[dt_col])
    except Exception:
        raise HTTPException(400, f"Cannot parse datetime column '{dt_col}' in {filename}.")
    raw = raw.rename(columns={dt_col: "Datetime"})

    if "MW" not in raw.columns:
        num_cols = raw.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            raise HTTPException(400, f"No numeric MW column found in {filename}.")
        raw = raw.rename(columns={num_cols[0]: "MW"})

    df = raw[["Datetime", "MW"]].dropna()
    df = df.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    return df


def _run_train(df: pd.DataFrame):
    try:
        metrics, importance, predictions, df_feat = train(df)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {traceback.format_exc()}")
    state.update(trained=True, metrics=metrics, importance=importance,
                 predictions=predictions, df_feat=df_feat)
    return metrics, importance, predictions


def _compute_profile(df: pd.DataFrame, name: str) -> dict:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    full = pd.date_range(df["Datetime"].min(), df["Datetime"].max(), freq="h")
    df = df.set_index("Datetime").reindex(full).rename_axis("Datetime").reset_index()
    df["MW"] = df["MW"].ffill()

    df["hour"]       = df["Datetime"].dt.hour
    df["dayofweek"]  = df["Datetime"].dt.dayofweek
    df["month"]      = df["Datetime"].dt.month
    df["year"]       = df["Datetime"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    years = df["year"].unique().tolist()
    us_hols = hol_lib.US(years=years)
    df["is_holiday"] = df["Datetime"].dt.date.astype(str).map(
        lambda d: 1 if pd.Timestamp(d) in us_hols else 0)

    avg   = float(df["MW"].mean())
    peak  = float(df["MW"].max())
    std   = float(df["MW"].std())
    total = float(df["MW"].sum())

    yearly  = df.groupby("year")["MW"].mean().reset_index().to_dict(orient="records")
    monthly = df.groupby("month")["MW"].mean().reset_index().to_dict(orient="records")
    hourly  = df.groupby("hour")["MW"].mean().reset_index().to_dict(orient="records")

    summer = float(df[df["month"].isin([6,7,8])]["MW"].mean())
    winter = float(df[df["month"].isin([12,1,2])]["MW"].mean())
    spring = float(df[df["month"].isin([3,4,5])]["MW"].mean())
    fall   = float(df[df["month"].isin([9,10,11])]["MW"].mean())

    weekday     = float(df[df["is_weekend"]==0]["MW"].mean())
    weekend     = float(df[df["is_weekend"]==1]["MW"].mean())
    holiday_avg = float(df[df["is_holiday"]==1]["MW"].mean()) if df["is_holiday"].sum() > 0 else 0.0

    yr_series = df.groupby("year")["MW"].mean()
    growth = float((yr_series.iloc[-1] - yr_series.iloc[0]) / yr_series.iloc[0] * 100) if len(yr_series) >= 2 else 0.0
    cv = float(std / avg * 100) if avg else 0.0
    peak_hour = int(df.groupby("hour")["MW"].mean().idxmax())

    return {
        "name": name,
        "date_min": str(df["Datetime"].min().date()),
        "date_max": str(df["Datetime"].max().date()),
        "total_rows": int(len(df)),
        "avg_mw": round(avg, 1),
        "peak_mw": round(peak, 1),
        "std_mw": round(std, 1),
        "total_mwh": round(total, 0),
        "growth_pct": round(growth, 2),
        "volatility_cv": round(cv, 2),
        "peak_hour": peak_hour,
        "seasonal": {"summer": round(summer,1), "winter": round(winter,1),
                     "spring": round(spring,1), "fall": round(fall,1)},
        "behavioral": {"weekday": round(weekday,1), "weekend": round(weekend,1),
                       "holiday": round(holiday_avg,1)},
        "yearly": yearly,
        "monthly": monthly,
        "hourly": hourly,
    }


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PJM Forecaster API running"}


@app.post("/train/demo")
def train_demo():
    df = generate_demo_data()
    metrics, importance, predictions = _run_train(df)
    return {"metrics": metrics, "importance": importance[:5], "predictions_count": len(predictions)}


@app.post("/train/upload")
async def train_upload(file: UploadFile = File(...)):
    content = await file.read()
    df = _parse_csv(content, file.filename)

    if len(df) < 1000:
        raise HTTPException(400, f"Too few rows ({len(df)}). Need at least 1000 hourly records.")

    metrics, importance, predictions = _run_train(df)

    min_date   = df["Datetime"].min()
    max_date   = df["Datetime"].max()
    has_2014   = (df["Datetime"].dt.year == 2014).any()
    test_split = "Full year 2014" if has_2014 else "Last 10%"

    return {
        "metrics": metrics,
        "importance": importance[:5],
        "predictions_count": len(predictions),
        "data_info": {
            "fileName": file.filename,
            "rows": len(df),
            "period": f"{min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}",
            "testSplit": test_split,
        }
    }


@app.get("/metrics")
def get_metrics():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    return state["metrics"]


@app.get("/importance")
def get_importance():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    return state["importance"]


@app.get("/predictions")
def get_predictions(limit: int = 3000):
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    return state["predictions"][:limit]


@app.get("/charts/heatmap")
def get_heatmap():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    df = state["df_feat"]
    return df.groupby(["dayofweek","hour"])["MW"].mean().reset_index().to_dict(orient="records")


@app.get("/charts/seasonal")
def get_seasonal():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    df = state["df_feat"]
    return {
        "summer": df[df["month"].isin([6,7,8])]["MW"].describe().to_dict(),
        "winter": df[df["month"].isin([12,1,2])]["MW"].describe().to_dict(),
    }


@app.get("/charts/yearly")
def get_yearly():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    return state["df_feat"].groupby("year")["MW"].mean().reset_index().to_dict(orient="records")


@app.get("/charts/holidays")
def get_holidays():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    df = state["df_feat"].copy()
    df["date"] = df["Datetime"].dt.date.astype(str)
    return df.groupby(["date","is_holiday"])["MW"].mean().reset_index().to_dict(orient="records")


@app.get("/analysis")
def get_analysis():
    if not state["trained"]: raise HTTPException(400, "Model not trained yet.")
    df = state["df_feat"].copy()

    hourly_avg  = df.groupby("hour")["MW"].mean().reset_index()
    monthly_avg = df.groupby("month")["MW"].mean().reset_index()
    yearly      = df.groupby("year")["MW"].mean().reset_index()

    peak_month_num = int(monthly_avg.loc[monthly_avg["MW"].idxmax(), "month"])
    low_month_num  = int(monthly_avg.loc[monthly_avg["MW"].idxmin(), "month"])
    MNAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    weekday_avg = float(df[df["is_weekend"]==0]["MW"].mean())
    weekend_avg = float(df[df["is_weekend"]==1]["MW"].mean())
    holiday_avg = float(df[df["is_holiday"]==1]["MW"].mean()) if df["is_holiday"].sum() > 0 else 0.0

    yr_vals = yearly["MW"]
    growth_pct = round(float((yr_vals.iloc[-1]-yr_vals.iloc[0])/yr_vals.iloc[0]*100),1) if len(yr_vals)>=2 else 0

    return {
        "overview": {
            "total_rows": int(len(df)),
            "date_min": str(df["Datetime"].min().date()),
            "date_max": str(df["Datetime"].max().date()),
            "avg_mw":  round(float(df["MW"].mean()),1),
            "peak_mw": round(float(df["MW"].max()),1),
            "min_mw":  round(float(df["MW"].min()),1),
            "peak_datetime": str(df.loc[df["MW"].idxmax(),"Datetime"]),
        },
        "hourly": {
            "peak_hour":   int(hourly_avg.loc[hourly_avg["MW"].idxmax(),"hour"]),
            "trough_hour": int(hourly_avg.loc[hourly_avg["MW"].idxmin(),"hour"]),
            "series": hourly_avg.to_dict(orient="records"),
        },
        "monthly": {
            "peak_month": MNAMES[peak_month_num],
            "low_month":  MNAMES[low_month_num],
            "series": monthly_avg.to_dict(orient="records"),
        },
        "seasonal": {
            "summer": round(float(df[df["month"].isin([6,7,8])]["MW"].mean()),1),
            "winter": round(float(df[df["month"].isin([12,1,2])]["MW"].mean()),1),
            "spring": round(float(df[df["month"].isin([3,4,5])]["MW"].mean()),1),
            "fall":   round(float(df[df["month"].isin([9,10,11])]["MW"].mean()),1),
        },
        "behavioral": {
            "weekday_avg": round(weekday_avg,1),
            "weekend_avg": round(weekend_avg,1),
            "weekend_drop_pct": round((weekday_avg-weekend_avg)/weekday_avg*100,1),
            "holiday_avg": round(holiday_avg,1),
            "holiday_drop_pct": round((weekday_avg-holiday_avg)/weekday_avg*100,1) if holiday_avg else 0,
        },
        "trend": {
            "growth_pct": growth_pct,
            "yearly": yearly.to_dict(orient="records"),
        },
    }


@app.post("/compare/upload")
async def compare_upload(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    try:
        content_a = await file_a.read()
        content_b = await file_b.read()

        df_a = _parse_csv(content_a, file_a.filename)
        df_b = _parse_csv(content_b, file_b.filename)

        name_a = file_a.filename.replace(".csv","").replace("_hourly","").replace("_"," ").strip()
        name_b = file_b.filename.replace(".csv","").replace("_hourly","").replace("_"," ").strip()

        profile_a = _compute_profile(df_a, name_a)
        profile_b = _compute_profile(df_b, name_b)

        compare_state.update(ready=True, profile_a=profile_a, profile_b=profile_b)
        return {"profile_a": profile_a, "profile_b": profile_b}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {traceback.format_exc()}")


@app.get("/compare/result")
def compare_result():
    if not compare_state["ready"]: raise HTTPException(400, "No comparison data yet.")
    return {"profile_a": compare_state["profile_a"], "profile_b": compare_state["profile_b"]}
