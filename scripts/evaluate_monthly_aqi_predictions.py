from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def summarize(g):
    y=g.actual_aqi.to_numpy(float); p=g.predicted_aqi.to_numpy(float)
    return pd.Series({"n_rows":len(g),"MAE":mean_absolute_error(y,p),
        "RMSE":np.sqrt(mean_squared_error(y,p)),"R2":r2_score(y,p) if len(y)>1 else np.nan})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("datasets/processed/monthly/model_results/predictions.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("datasets/processed/monthly/model_results"))
    a=p.parse_args()
    d=pd.read_csv(a.input)
    required={"model","split","borough","target_month","actual_aqi","predicted_aqi"}
    if required-set(d.columns): raise ValueError(f"Missing columns: {sorted(required-set(d.columns))}")
    d["target_month"]=pd.to_datetime(d.target_month,errors="coerce").dt.to_period("M").dt.to_timestamp()
    for c in ("actual_aqi","predicted_aqi"): d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["target_month","actual_aqi","predicted_aqi"])
    if d.empty: raise ValueError("No valid prediction rows.")
    d["absolute_error"]=(d.actual_aqi-d.predicted_aqi).abs()
    d["signed_error"]=d.predicted_aqi-d.actual_aqi
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,groups in {
        "evaluation_summary.csv":["model","split"],
        "evaluation_by_borough.csv":["model","split","borough"],
        "evaluation_by_month.csv":["model","split","target_month"],
    }.items():
        d.groupby(groups,dropna=False).apply(summarize,include_groups=False).reset_index().to_csv(a.output_dir/name,index=False)
    d.assign(month_of_year=d.target_month.dt.month).groupby(
        ["model","split","month_of_year"],dropna=False).apply(summarize,include_groups=False).reset_index().to_csv(
            a.output_dir/"evaluation_by_month_of_year.csv",index=False)
    d.to_csv(a.output_dir/"evaluation_predictions.csv",index=False)
    print(pd.read_csv(a.output_dir/"evaluation_summary.csv").to_string(index=False))
    print(f"Evaluation files saved to {a.output_dir}")

if __name__=="__main__": main()
