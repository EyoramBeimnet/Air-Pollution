"""Create recursive monthly XGBoost AQI forecasts through December 2030.

This is a forecast-delivery script, not a replacement for the existing monthly
training experiment. It uses that experiment's three approved inputs: lag 1,
lag 12, and calendar month. After December 2025, earlier predictions are used
as the lag inputs; no future AQI observations are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets/processed/monthly/monthly_aqi_model_STAGING.csv"
OUT = ROOT / "datasets/processed/monthly/model_results/monthly_aqi_forecast_2026_2030.csv"
META = ROOT / "datasets/processed/monthly/model_results/monthly_aqi_forecast_2026_2030_metadata.json"
BACKTEST = ROOT / "datasets/processed/monthly/model_results/monthly_recursive_backtest.csv"
FIPS = {"36005", "36047", "36061", "36081", "36085"}
FEATURES = ["aqi_lag_1", "aqi_lag_12", "target_month_number"]
ORIGIN = pd.Timestamp("2025-12-01")
FORECAST_MONTHS = pd.date_range("2026-01-01", "2030-12-01", freq="MS")


def model() -> XGBRegressor:
    return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.03,
        min_child_weight=3, subsample=.8, colsample_bytree=.8, reg_lambda=5,
        objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=-1)


def load() -> pd.DataFrame:
    d = pd.read_csv(INPUT, dtype={"county_fips": "string"})
    required = {"county_fips", "borough", "target_month", "aqi_median_daily", *FEATURES}
    if required - set(d.columns): raise ValueError(f"Missing monthly columns: {sorted(required-set(d.columns))}")
    d["county_fips"] = d.county_fips.astype("string").str.zfill(5)
    d["target_month"] = pd.to_datetime(d.target_month).dt.to_period("M").dt.to_timestamp()
    for c in ["aqi_median_daily", *FEATURES]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.county_fips.isin(FIPS)].dropna(subset=["aqi_median_daily", "aqi_lag_1", "aqi_lag_12"]).copy()
    if d.duplicated(["county_fips", "target_month"]).any(): raise ValueError("Duplicate monthly keys.")
    return d.sort_values(["target_month", "county_fips"])


def fit(frame: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    x = imp.fit_transform(frame[FEATURES])
    m = model(); m.fit(x, frame.aqi_median_daily.to_numpy(float))
    return imp, m


def recursive(frame: pd.DataFrame, train_end: pd.Timestamp, months: pd.DatetimeIndex) -> pd.DataFrame:
    train = frame[frame.target_month <= train_end]
    imp, m = fit(train)
    history = {(str(r.county_fips), pd.Timestamp(r.target_month)): float(r.aqi_median_daily) for r in train.itertuples()}
    boroughs = train.drop_duplicates("county_fips").set_index("county_fips").borough.to_dict()
    rows = []
    for month in months:
        for fips in sorted(FIPS):
            lag1, lag12 = history.get((fips, month - pd.offsets.MonthBegin(1))), history.get((fips, month - pd.DateOffset(years=1)))
            if lag1 is None or lag12 is None: raise ValueError(f"Missing recursive lag for {fips}/{month:%Y-%m}")
            features = pd.DataFrame([[lag1, lag12, month.month]], columns=FEATURES)
            value = float(m.predict(imp.transform(features))[0])
            if not np.isfinite(value) or value < 0: raise ValueError("Invalid monthly model prediction.")
            history[(fips, month)] = value
            rows.append({"fips":fips, "borough":boroughs[fips], "month":month.strftime("%Y-%m"),
                         "aqi":round(value, 3), "forecast":True,
                         "metric":"monthly_median_daily_aqi", "forecast_method":"recursive_lag_xgboost"})
    return pd.DataFrame(rows)


def backtest(frame: pd.DataFrame) -> pd.DataFrame:
    # Two origins support the full 60-month horizon while retaining enough history.
    rows=[]; actual={(str(r.county_fips), pd.Timestamp(r.target_month)):float(r.aqi_median_daily) for r in frame.itertuples()}
    for origin in (pd.Timestamp("2020-12-01"), pd.Timestamp("2021-12-01")):
        months=pd.date_range(origin + pd.offsets.MonthBegin(1), periods=60, freq="MS")
        pred=recursive(frame, origin, months)
        for r in pred.itertuples():
            month=pd.Timestamp(r.month+"-01"); truth=actual.get((r.fips, month))
            if truth is not None:
                h=(month.year-origin.year)*12 + month.month-origin.month
                baseline=actual[(r.fips, origin)]
                rows.extend([{"model":"XGBoost recursive","horizon_months":h,"actual":truth,"predicted":r.aqi},
                             {"model":"Persistence baseline","horizon_months":h,"actual":truth,"predicted":baseline}])
    report=[]
    for (name,h), g in pd.DataFrame(rows).groupby(["model","horizon_months"]):
        y,p=g.actual.to_numpy(float),g.predicted.to_numpy(float)
        report.append({"model":name,"horizon_months":h,"n_rows":len(g),"MAE":mean_absolute_error(y,p),"RMSE":np.sqrt(mean_squared_error(y,p))})
    return pd.DataFrame(report).sort_values(["horizon_months","model"])


def main() -> None:
    d=load(); forecast=recursive(d, ORIGIN, FORECAST_MONTHS)
    if len(forecast)!=300 or forecast.duplicated(["fips","month"]).any(): raise ValueError("Forecast coverage validation failed.")
    OUT.parent.mkdir(parents=True,exist_ok=True); forecast.to_csv(OUT,index=False); backtest(d).to_csv(BACKTEST,index=False)
    META.write_text(json.dumps({"metric":"monthly_median_daily_aqi","model":"XGBoost from train_monthly_aqi_models.py","features":FEATURES,"training_months":[str(d.target_month.min())[:7],str(ORIGIN)[:7]],"forecast_origin":"2025-12","forecast_months":["2026-01","2030-12"],"method":"Recursive; predicted values become lag inputs after the observed history ends.","limitations":["Multi-step error accumulates; see monthly_recursive_backtest.csv.","Point forecasts only; no intervals." ]},indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {len(forecast)} monthly forecast rows to {OUT}")

if __name__ == "__main__": main()
