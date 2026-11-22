"""Build 2026--2030 annual AQI forecasts using the approved fixed-2025 scenario.

2026 uses observed 2025 AQI, CAMPD NOx, temperature and precipitation. For
2027--2030, predicted prior-year AQI is fed back recursively while the three
external features are intentionally held at each borough's observed 2025 value.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_aqi_models import ABLATION_FEATURES, FIPS_COLUMN, TARGET_COLUMN, YEAR_COLUMN, load_staging_data, build_models

ROOT=Path(__file__).resolve().parents[1]; MASTER=ROOT/"datasets/processed/master_borough_year_2014_2025.csv"
FORECAST_2026=ROOT/"datasets/processed/model_results/annual_aqi_forecast_2026.csv"
OUT=ROOT/"datasets/processed/model_results/annual_aqi_forecast_2026_2030.csv"; META=ROOT/"datasets/processed/model_results/annual_aqi_forecast_2026_2030_metadata.json"; BACKTEST=ROOT/"datasets/processed/model_results/annual_recursive_scenario_backtest.csv"
FIPS={"36005","36047","36061","36081","36085"}; FEATURES=ABLATION_FEATURES["Ablation weather"]

def train_data():
 d=load_staging_data(ROOT/"datasets/processed/next_year_aqi_model_STAGING.csv"); d=d[d[YEAR_COLUMN].between(2014,2024)].copy()
 if len(d)!=55 or set(d[FIPS_COLUMN])!=FIPS: raise ValueError("Expected 55 annual labelled rows.")
 return d

def fit(d):
 imp=SimpleImputer(strategy="median"); x=imp.fit_transform(d[FEATURES]); m=build_models()["XGBoost"]; m.fit(x,d[TARGET_COLUMN].to_numpy(float)); return imp,m

def scenario_forecast(d, feature_rows, years=range(2026,2031)):
 imp,m=fit(d); rows=[]
 legacy=pd.read_csv(FORECAST_2026,dtype={"fips":"string"})
 if len(legacy)!=5 or set(legacy.fips.astype("string").str.zfill(5))!=FIPS or set(legacy.year)!={2026}: raise ValueError("Existing 2026 forecast artifact is incomplete.")
 legacy_by_fips={str(r.fips).zfill(5):float(r.aqi) for r in legacy.itertuples()}
 for r in feature_rows.itertuples():
  previous=float(r.last_observed_median_aqi)
  for year in years:
   x=imp.transform(pd.DataFrame([[previous,float(r.campd_total_nox_mass_tons),float(r.weather_temp_mean_c),float(r.weather_precip_total_mm)]],columns=FEATURES))
   value=legacy_by_fips[str(r.county_fips)] if year==2026 else float(m.predict(x)[0])
   if not np.isfinite(value) or value<0: raise ValueError("Invalid annual scenario prediction.")
   rows.append({"fips":str(r.county_fips),"borough":r.borough,"year":year,"aqi":round(value,3),"forecast":True,"metric":"annual_median_daily_aqi","forecast_method":"observed_2025_inputs" if year==2026 else "recursive_fixed_2025_external_features_scenario"})
   previous=value
 return pd.DataFrame(rows)

def backtest(d):
 rows=[]
 # origin feature year t predicts t+1 onward, holding t's external values fixed.
 for origin in range(2018,2021):
  train=d[d.feature_year<origin]; imp,m=fit(train)
  for fips,g in d[d.feature_year.eq(origin)].groupby(FIPS_COLUMN):
   base=g.iloc[0]; previous=float(base.last_observed_median_aqi)
   fixed=[float(base.campd_total_nox_mass_tons),float(base.weather_temp_mean_c),float(base.weather_precip_total_mm)]
   for h in range(1,6):
    target=origin+h; actual=d[(d[FIPS_COLUMN]==fips)&(d.forecast_year==target)]
    if actual.empty: break
    values=pd.DataFrame([[previous,*fixed]],columns=FEATURES)
    pred=float(m.predict(imp.transform(values))[0]); truth=float(actual.iloc[0][TARGET_COLUMN])
    rows += [{"model":"XGBoost fixed-feature scenario","horizon_years":h,"actual":truth,"predicted":pred},{"model":"Persistence baseline","horizon_years":h,"actual":truth,"predicted":float(base.last_observed_median_aqi)}]
    previous=pred
 report=[]
 for (model,h),g in pd.DataFrame(rows).groupby(["model","horizon_years"]):
  y,p=g.actual.to_numpy(float),g.predicted.to_numpy(float); report.append({"model":model,"horizon_years":h,"n_rows":len(g),"MAE":mean_absolute_error(y,p),"RMSE":np.sqrt(mean_squared_error(y,p))})
 return pd.DataFrame(report).sort_values(["horizon_years","model"])

def main():
 d=train_data(); master=pd.read_csv(MASTER,dtype={"county_fips":"string"}); f=master[master.year.eq(2025)].copy(); f["county_fips"]=f.county_fips.astype("string").str.zfill(5); f["last_observed_median_aqi"]=f.aqi_median
 need={"county_fips","borough","last_observed_median_aqi",*FEATURES[1:]}
 if need-set(f.columns) or len(f)!=5 or set(f.county_fips)!=FIPS or f[list(need-{"county_fips","borough"})].isna().any().any(): raise ValueError("Incomplete observed 2025 annual inputs.")
 forecast=scenario_forecast(d,f); OUT.parent.mkdir(parents=True,exist_ok=True); forecast.to_csv(OUT,index=False); backtest(d).to_csv(BACKTEST,index=False)
 META.write_text(json.dumps({"metric":"annual_median_daily_aqi","model":"Weather-enhanced XGBoost from train_aqi_models.py","features":FEATURES,"training_feature_years":[2014,2024],"forecast_years":[2026,2030],"forecast_origin":"observed 2025 inputs","method":"2026 uses observed 2025 features; 2027--2030 recursively use predicted AQI while CAMPD NOx, temperature, and precipitation remain fixed at their observed 2025 values.","limitations":["2027--2030 are conditional scenario projections, not forecasts with observed future external inputs.","Multi-step accuracy is reported separately in annual_recursive_scenario_backtest.csv.","Point forecasts only; no intervals."]},indent=2)+"\n",encoding="utf-8")
 print(f"Wrote {len(forecast)} annual forecast rows to {OUT}")
if __name__=="__main__": main()
