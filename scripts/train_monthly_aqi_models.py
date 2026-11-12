from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def score(y, pred):
    return {"MAE":mean_absolute_error(y,pred), "RMSE":np.sqrt(mean_squared_error(y,pred)),
            "R2":r2_score(y,pred) if len(y)>1 else np.nan}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("datasets/processed/monthly/monthly_aqi_model_STAGING.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("datasets/processed/monthly/model_results"))
    p.add_argument("--train-fraction",type=float,default=.70)
    p.add_argument("--validation-fraction",type=float,default=.15)
    a=p.parse_args()
    d=pd.read_csv(a.input,dtype={"county_fips":"string"})
    required={"county_fips","borough","target_month","aqi_median_daily","aqi_lag_1"}
    if required-set(d.columns): raise ValueError(f"Missing columns: {sorted(required-set(d.columns))}")
    d["target_month"]=pd.to_datetime(d.target_month).dt.to_period("M").dt.to_timestamp()
    d["county_fips"]=d.county_fips.astype("string").str.zfill(5)
    d["aqi_median_daily"]=pd.to_numeric(d.aqi_median_daily,errors="coerce")
    d=d.dropna(subset=["target_month","aqi_median_daily","aqi_lag_1"]).sort_values(["target_month","county_fips"])
    if d.duplicated(["county_fips","target_month"]).any(): raise ValueError("Duplicate county/month rows.")
    if a.train_fraction<=0 or a.validation_fraction<=0 or a.train_fraction+a.validation_fraction>=1: raise ValueError("Invalid split fractions.")
    excluded={"county_fips","borough","target_month","issue_month","aqi_median_daily","aqi_mean_daily",
              "aqi_days_observed","calendar_days_in_month","aqi_coverage_pct","meets_coverage","target_year"}
    features=[c for c in d.columns if c not in excluded and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().any()]
    months=sorted(d.target_month.unique())
    if len(months)<18: raise ValueError(f"Only {len(months)} target months; need at least 18.")
    i=max(1,int(len(months)*a.train_fraction)); j=min(len(months)-1,max(i+1,int(len(months)*(a.train_fraction+a.validation_fraction))))
    splits={"train":d[d.target_month.isin(months[:i])].copy(),
            "validation":d[d.target_month.isin(months[i:j])].copy(),
            "test":d[d.target_month.isin(months[j:])].copy()}
    if any(x.empty for x in splits.values()): raise ValueError("A split is empty.")
    models={"Random Forest":RandomForestRegressor(n_estimators=300,max_depth=5,min_samples_leaf=2,random_state=42,n_jobs=-1),
            "XGBoost":XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.03,min_child_weight=3,
                subsample=.8,colsample_bytree=.8,reg_lambda=5,objective="reg:squarederror",eval_metric="rmse",random_state=42,n_jobs=-1)}
    pred_rows=[]; metric_rows=[]; train=splits["train"]
    for split in ("validation","test"):
        frame=splits[split]; pred=frame.aqi_lag_1.to_numpy(float)
        metric_rows.append({"model":"Persistence baseline","split":split,"n_rows":len(frame),**score(frame.aqi_median_daily,pred)})
        for k,(_,r) in enumerate(frame.iterrows()):
            pred_rows.append({"model":"Persistence baseline","split":split,"county_fips":r.county_fips,
                "borough":r.borough,"target_month":r.target_month,"actual_aqi":r.aqi_median_daily,
                "predicted_aqi":pred[k],"training_rows":len(train)})
    for name,est in models.items():
        model=make_pipeline(SimpleImputer(strategy="median"),est)
        model.fit(train[features],train.aqi_median_daily)
        for split,frame in splits.items():
            pred=model.predict(frame[features])
            metric_rows.append({"model":name,"split":split,"n_rows":len(frame),**score(frame.aqi_median_daily,pred)})
            for k,(_,r) in enumerate(frame.iterrows()):
                pred_rows.append({"model":name,"split":split,"county_fips":r.county_fips,
                    "borough":r.borough,"target_month":r.target_month,"actual_aqi":r.aqi_median_daily,
                    "predicted_aqi":pred[k],"training_rows":len(train)})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(a.output_dir/"predictions.csv",index=False)
    pd.DataFrame(metric_rows).to_csv(a.output_dir/"training_metrics.csv",index=False)
    pd.DataFrame({"feature":features}).to_csv(a.output_dir/"features_used.csv",index=False)
    for name,frame in splits.items(): print(f"{name}: {frame.target_month.min():%Y-%m} to {frame.target_month.max():%Y-%m}; {len(frame)} rows")
    print(f"Saved model outputs to {a.output_dir}")

if __name__=="__main__": main()
