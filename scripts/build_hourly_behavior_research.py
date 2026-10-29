from __future__ import annotations
import hashlib, json, math, resource, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw"
CANDLE_DIR=RAW/"market/bitmex/candle_v1"
AUDIT=ROOT/"data/audit/candle_v1"
OUT=ROOT/"data/processed/research/hourly_behavior_v2"
CHUNK=50000
BOOTSTRAPS=5000
BLOCK=3
SEED=20261020
ZERO_UUID="00000000-0000-0000-0000-000000000000"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str)+"\n")

def load_market():
    verification=json.loads((AUDIT/"candle_source_verification.json").read_text())
    files=sorted(CANDLE_DIR.glob("*.parquet"))
    market=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    market["candle_end_utc"]=pd.to_datetime(market["candle_end_utc"],utc=True)
    market=market.sort_values("candle_end_utc").reset_index(drop=True)
    if market.candle_end_utc.duplicated().any(): raise ValueError("duplicate hourly candle ends")
    if not (market.candle_end_utc.diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError("hourly candle gap or duplicate")
    for col in ("open","high","low","close","volume"):
        market[col]=pd.to_numeric(market[col],errors="raise").astype("float64")
    if ((market[["open","high","low","close"]]<=0).any().any() or
        (market.volume<0).any() or (market.high<market.low).any() or
        (~market.close.between(market.low,market.high)).any()):
        raise ValueError("hourly OHLCV validation failed")
    px=market.close
    hourly_ret=px.pct_change()
    market["return_24h"]=px.div(px.shift(24)).sub(1)
    market["return_7d"]=px.div(px.shift(168)).sub(1)
    market["return_30d"]=px.div(px.shift(720)).sub(1)
    market["volatility_30d_ann"]=hourly_ret.rolling(720,min_periods=720).std(ddof=1)*math.sqrt(24*365)
    direction=np.select([market.return_30d>=.10,market.return_30d<=-.10],["BULL","BEAR"],default="RANGE")
    vol=np.where(market.volatility_30d_ann>=.80,"HIGH_VOL","NORMAL_VOL")
    market["regime"]=np.where(market.return_30d.isna()|market.volatility_30d_ann.isna(),"WARMUP_UNKNOWN",
                              np.char.add(np.char.add(direction,"_"),vol))
    market["candle_start_utc"]=market.candle_end_utc-pd.Timedelta(hours=1)
    market["year"]=market.candle_start_utc.dt.year.astype("int16")
    market["month"]=market.candle_end_utc.dt.strftime("%Y-%m")
    market["hour"]=1
    if int(verification.get("missing_hours",-1))!=0: raise ValueError("existing verification does not report zero missing hours")
    return market,files,verification

def main():
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    t0=time.time()
    market,candle_files,candle_verification=load_market()
    ends_ns=market.candle_end_utc.astype("int64").to_numpy()
    feat_cols=["return_24h","return_7d","return_30d","volatility_30d_ann","regime","close"]
    candle_features=market[feat_cols].to_numpy()
    raw_files=sorted(RAW.glob("aoa-execution-*.csv"))
    if len(raw_files)!=4: raise ValueError(f"expected four execution CSVs, found {len(raw_files)}")
    profile=json.loads((ROOT/"data/audit/account_20261012/csv_profile.json").read_text())
    expected={Path(x["path"]).name:x["sha256"] for x in profile["files"]}
    hashes={}
    for p in raw_files:
        got=sha256(p)
        if expected.get(p.name)!=got: raise ValueError(f"source hash mismatch: {p.name}")
        hashes[p.name]=got
    total=0; per_file={}; sizes={}; missing=Counter(); joins=Counter()
    mismatch=0; event_min=event_max=xbt_min=xbt_max=None; part_no=0
    for src in raw_files:
        sizes[src.name]=src.stat().st_size; line=0; nfile=0
        for raw in pd.read_csv(src,chunksize=CHUNK,dtype=str,keep_default_na=False,encoding="utf-8-sig"):
            n=len(raw); line0=line; line+=n; nfile+=n; total+=n
            raw["source_file"]=src.name
            raw["source_file_sha256"]=hashes[src.name]
            raw["source_record_number"]=np.arange(line0+1,line+1,dtype="int64")
            event=pd.to_datetime(raw.transacttime.replace("",pd.NA),utc=True,errors="coerce")
            proc=pd.to_datetime(raw.timestamp.replace("",pd.NA),utc=True,errors="coerce")
            missing["transacttime"]+=int(event.isna().sum()); missing["timestamp"]+=int(proc.isna().sum())
            both=event.notna()&proc.notna()
            mismatch+=int((event[both]!=proc[both]).sum())
            if event.notna().any():
                event_min=event.min() if event_min is None else min(event_min,event.min())
                event_max=event.max() if event_max is None else max(event_max,event.max())
            valid=event.notna().to_numpy(); evns=event.dt.as_unit("ns").astype("int64").to_numpy()
            idx=np.full(n,-1,dtype="int64")
            idx[valid]=np.searchsorted(ends_ns,evns[valid],side="left")-1
            isx=raw.symbol.eq("XBTUSD").to_numpy()
            joined=isx&valid&(idx>=0)
            raw["event_time_assumed_utc"]=event
            raw["timestamp_provenance"]="NAIVE_AS_UTC_EXPLORATORY"
            raw["market_join_status"]=np.where(~isx,"UNSUPPORTED_SYMBOL",
                np.where(~valid,"MISSING_EVENT_TIME",np.where(idx<0,"NO_PRIOR_COMPLETED_CANDLE","MATCHED")))
            asof=np.full(n,-1,dtype="int64"); asof[joined]=ends_ns[idx[joined]]
            raw["market_asof_candle_end_utc"]=pd.to_datetime(pd.Series(asof,index=raw.index).where(asof>=0),utc=True)
            raw["market_asof_close"]=np.nan
            for c in ("return_24h","return_7d","return_30d","volatility_30d_ann"): raw["market_"+c]=np.nan
            raw["market_regime"]="NOT_JOINED"
            if joined.any():
                ii=idx[joined]
                raw.loc[joined,"market_asof_close"]=candle_features[ii,5].astype("float64")
                for j,c in enumerate(("return_24h","return_7d","return_30d","volatility_30d_ann")):
                    raw.loc[joined,"market_"+c]=candle_features[ii,j].astype("float64")
                raw.loc[joined,"market_regime"]=candle_features[ii,4]
            joins.update(raw.market_join_status.value_counts().to_dict())
            xt=raw.exectype.eq("Trade")&raw.symbol.eq("XBTUSD")
            if xt.any():
                tm=event[xt]
                xbt_min=tm.min() if xbt_min is None else min(xbt_min,tm.min())
                xbt_max=tm.max() if xbt_max is None else max(xbt_max,tm.max())
            yr=event.dt.year.fillna(0).astype("int16")
            mo=event.dt.month.fillna(0).astype("int8")
            for (year,month),part in raw.groupby([yr,mo],sort=True,dropna=False):
                year=int(year); month=int(month)
                dest=OUT/("year=unknown/month=unknown" if not year or not month else f"year={year:04d}/month={month:02d}")
                dest.mkdir(parents=True,exist_ok=True); part_no+=1
                part.to_parquet(dest/f"part-{part_no:06d}.parquet",index=False,compression="zstd")
        per_file[src.name]=nfile

    files=sorted(OUT.glob("year=*/month=*/*.parquet"))
    cols=["execid","event_time_assumed_utc","symbol","exectype","side","lastqty","lastpx","lastliquidityind",
          "orderid","ordstatus","leavesqty","orderqty","market_regime","market_join_status"]
    data=pd.concat([pd.read_parquet(p,columns=cols) for p in files],ignore_index=True)
    xbt=data[(data.symbol=="XBTUSD")&(data.exectype=="Trade")].copy()
    xbt["qty"]=pd.to_numeric(xbt.lastqty,errors="coerce")
    xbt["day"]=xbt.event_time_assumed_utc.dt.strftime("%Y-%m-%d")
    xbt["month"]=xbt.event_time_assumed_utc.dt.strftime("%Y-%m")
    xbt["year"]=xbt.event_time_assumed_utc.dt.year
    xbt["buy"]=xbt.side.eq("Buy"); xbt["sell"]=xbt.side.eq("Sell")
    xbt["maker"]=xbt.lastliquidityind.eq("AddedLiquidity")
    xbt["taker"]=xbt.lastliquidityind.eq("RemovedLiquidity")
    xbt["unknown_liq"]=~(xbt.maker|xbt.taker)
    xbt["eligible_regime"]=~xbt.market_regime.isin(["NOT_JOINED","WARMUP_UNKNOWN"])
    reg=xbt[xbt.eligible_regime].groupby("market_regime").agg(
        trade_rows=("side","size"),buy_rows=("buy","sum"),sell_rows=("sell","sum"),contracts=("qty","sum"),
        median_fill_contracts=("qty","median"),p90_fill_contracts=("qty",lambda s:s.quantile(.90)),
        maker_rows=("maker","sum"),taker_rows=("taker","sum"),unknown_liquidity_rows=("unknown_liq","sum"),
        active_days=("day","nunique"),active_months=("month","nunique")).reset_index()
    reg["maker_share_known"]=reg.maker_rows/(reg.maker_rows+reg.taker_rows)
    reg_hours=market[market.regime.ne("WARMUP_UNKNOWN")].groupby("regime").hour.sum()
    reg["exposure_hours"]=reg.market_regime.map(reg_hours).astype("int64")
    reg["trades_per_1000_regime_hours"]=reg.trade_rows/reg.exposure_hours*1000
    reg["buy_share"]=reg.buy_rows/reg.trade_rows
    reg.to_csv(OUT/"by_market_regime.csv",index=False)
    annual=xbt.groupby("year").agg(trade_rows=("side","size"),buy_rows=("buy","sum"),sell_rows=("sell","sum"),
        contracts=("qty","sum"),median_fill_contracts=("qty","median"),p90_fill_contracts=("qty",lambda s:s.quantile(.90)),
        maker_rows=("maker","sum"),taker_rows=("taker","sum"),unknown_liquidity_rows=("unknown_liq","sum"),
        active_days=("day","nunique"),active_months=("month","nunique")).reset_index()
    annual["maker_share_known"]=annual.maker_rows/(annual.maker_rows+annual.taker_rows)
    annual["buy_share"]=annual.buy_rows/annual.trade_rows
    annual.to_csv(OUT/"by_year.csv",index=False)
    yrreg=xbt[xbt.eligible_regime].groupby(["year","market_regime"]).agg(
        trade_rows=("side","size"),buy_rows=("buy","sum"),sell_rows=("sell","sum"),contracts=("qty","sum"),
        median_fill_contracts=("qty","median"),maker_rows=("maker","sum"),taker_rows=("taker","sum"),
        active_days=("day","nunique")).reset_index()
    yrreg["maker_share_known"]=yrreg.maker_rows/(yrreg.maker_rows+yrreg.taker_rows)
    yrreg.to_csv(OUT/"by_year_and_regime.csv",index=False)
    market["month"]=market.candle_end_utc.dt.strftime("%Y-%m")
    exposure=market[market.regime.ne("WARMUP_UNKNOWN")].groupby(["month","regime"]).agg(exposure_hours=("hour","sum")).reset_index()
    exm=xbt[xbt.eligible_regime].groupby(["month","market_regime"]).agg(
        trade_rows=("side","size"),buy_rows=("buy","sum"),maker_rows=("maker","sum"),taker_rows=("taker","sum")).reset_index().rename(columns={"market_regime":"regime"})
    monthly=exposure.merge(exm,on=["month","regime"],how="left").fillna({"trade_rows":0,"buy_rows":0,"maker_rows":0,"taker_rows":0})
    monthly["trades_per_1000_regime_hours"]=monthly.trade_rows/monthly.exposure_hours*1000
    monthly["buy_share"]=np.where(monthly.trade_rows>0,monthly.buy_rows/monthly.trade_rows,np.nan)
    monthly["maker_share_known"]=np.where(monthly.maker_rows+monthly.taker_rows>0,monthly.maker_rows/(monthly.maker_rows+monthly.taker_rows),np.nan)
    monthly.to_csv(OUT/"monthly_regime_activity.csv",index=False)
    # Moving 3-month block bootstrap for high vs normal volatility execution intensity.
    vm=monthly.assign(vol_group=monthly.regime.map(lambda r: "HIGH_VOL" if r.endswith("_HIGH_VOL") else ("NORMAL_VOL" if r.endswith("_NORMAL_VOL") else "UNKNOWN"))).groupby(["month","vol_group"]).agg(exposure=("exposure_hours","sum"),trades=("trade_rows","sum")).reset_index()
    wide=vm.pivot(index="month",columns="vol_group",values=["exposure","trades"]).sort_index()
    months=pd.period_range("2018-03","2021-12",freq="M").astype(str)
    wide=wide.reindex(months).fillna(0); rng=np.random.default_rng(SEED); n=len(wide); diffs=[]
    for _ in range(BOOTSTRAPS):
        starts=rng.integers(0,n,size=math.ceil(n/BLOCK))
        ids=np.concatenate([(np.arange(s,s+BLOCK)%n) for s in starts])[:n]
        smp=wide.iloc[ids]
        he=float(smp[("exposure","HIGH_VOL")].sum()); ne=float(smp[("exposure","NORMAL_VOL")].sum())
        if he and ne: diffs.append(float(smp[("trades","HIGH_VOL")].sum()/he*1000-smp[("trades","NORMAL_VOL")].sum()/ne*1000))
    hr=float(wide[("trades","HIGH_VOL")].sum()/wide[("exposure","HIGH_VOL")].sum()*1000)
    nr=float(wide[("trades","NORMAL_VOL")].sum()/wide[("exposure","NORMAL_VOL")].sum()*1000)
    boot={"method":"circular 3-month moving-block bootstrap; calendar months resampled as time blocks",
          "replicates":len(diffs),"seed":SEED,"block_months":BLOCK,"high_vol_trades_per_1000_hours":hr,
          "normal_vol_trades_per_1000_hours":nr,"difference":hr-nr,
          "difference_95pct_ci":[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))] if diffs else None,
          "interpretation":"descriptive association, not causal; thresholds fixed in advance"}
    write_json(OUT/"bootstrap_volatility_activity.json",boot)

    known=xbt.orderid.ne("")&xbt.orderid.ne(ZERO_UUID)
    xo=xbt[known].sort_values(["orderid","event_time_assumed_utc"],kind="stable")
    orders=xo.groupby("orderid",sort=False).agg(fill_rows=("qty","size"),fill_contracts=("qty","sum"),
        first_fill=("event_time_assumed_utc","min"),last_fill=("event_time_assumed_utc","max"),
        last_leaves=("leavesqty","last"),final_status=("ordstatus","last"),
        maker_fills=("maker","sum"),taker_fills=("taker","sum"),regime=("market_regime","first")).reset_index()
    orders["multi_fill"]=orders.fill_rows>1
    orders["observed_partial"]=orders.final_status.eq("PartiallyFilled")|pd.to_numeric(orders.last_leaves,errors="coerce").fillna(0).gt(0)
    orders["duration_seconds"]=(orders.last_fill-orders.first_fill).dt.total_seconds()
    ob=orders.groupby("regime").agg(order_ids=("orderid","size"),multi_fill_orders=("multi_fill","sum"),
        observed_partial_state_orders=("observed_partial","sum"),median_fill_rows=("fill_rows","median"),
        median_fill_duration_seconds=("duration_seconds","median"),maker_fills=("maker_fills","sum"),
        taker_fills=("taker_fills","sum")).reset_index()
    ob["maker_fill_share"]=ob.maker_fills/(ob.maker_fills+ob.taker_fills)
    ob.to_csv(OUT/"order_behavior_by_regime.csv",index=False)
    order_summary={"xbtusd_trade_rows":int(len(xbt)),"known_orderid_trade_rows":int(known.sum()),
        "missing_or_zero_orderid_trade_rows":int((~known).sum()),"distinct_nonzero_orderids":int(xo.orderid.nunique()),
        "multi_fill_orders":int(orders.multi_fill.sum()),"orders_with_observed_partial_state":int(orders.observed_partial.sum()),
        "final_status_counts":{str(k):int(v) for k,v in orders.final_status.value_counts(dropna=False).items()},
        "trade_rows_with_cancel_status":int(xbt.ordstatus.isin(["Canceled","CanceledByUser","CanceledByExchange"]).sum()),
        "limitation":"execution-only export; unfilled orders and cancellations without a fill cannot be measured"}
    write_json(OUT/"order_behavior_summary.json",order_summary)

    pc=pd.read_parquet(AUDIT/"position_changes.parquet")
    market_tags=data[["execid","event_time_assumed_utc","market_regime"]].drop_duplicates("execid")
    pc=pc.merge(market_tags,on="execid",how="left",validate="one_to_one")
    pc["year"]=pc.event_time_assumed_utc.dt.year
    state=pc.position_before_qty.notna()&pc.position_after_qty.notna()
    b=pc.loc[state,"position_before_qty"].astype(float); a=pc.loc[state,"position_after_qty"].astype(float)
    pos={"artifact":"data/audit/candle_v1/position_changes.parquet","trade_rows":int(len(pc)),
         "candidate_state_rows":int(state.sum()),"unknown_state_rows":int((~state).sum()),
         "scale_in_rows":int((a.abs()>b.abs()).sum()),"reduce_rows":int((a.abs()<b.abs()).sum()),
         "flat_after_rows":int((a==0).sum()),"candidate_reversal_rows":int(((a*b)<0).sum()),
         "status":"EXPLORATORY_RECONSTRUCTION_ONLY",
         "caveat":"Not verified positions: configs/account_evidence.json leaves position evidence UNKNOWN; existing reconstruction depends on inferred funding anchors and assumed timestamps."}
    write_json(OUT/"candidate_position_transition_summary.json",pos)
    pc_known=pc[state].copy()
    pc_known["scale_in"]=pc_known.position_after_qty.abs()>pc_known.position_before_qty.abs()
    pc_known["reduce"]=pc_known.position_after_qty.abs()<pc_known.position_before_qty.abs()
    pc_known["flat_after"]=pc_known.position_after_qty.eq(0)
    pc_known["reversal"]=(pc_known.position_after_qty*pc_known.position_before_qty)<0
    pc_summary=pc_known.groupby(["year","market_regime"]).agg(candidate_rows=("execid","size"),
        scale_in_rows=("scale_in","sum"),reduce_rows=("reduce","sum"),flat_after_rows=("flat_after","sum"),
        reversal_rows=("reversal","sum")).reset_index()
    pc_summary["status"]="EXPLORATORY_RECONSTRUCTION_ONLY"
    pc_summary.to_csv(OUT/"candidate_position_transitions_by_year_regime.csv",index=False)

    annual_market=market.groupby("year").agg(hours=("hour","sum"),open_first=("open","first"),close_last=("close","last"),
        mean_hourly_volume=("volume","mean"),median_hourly_volume=("volume","median"),
        mean_30d_ann_vol=("volatility_30d_ann","mean"),
        bull_hours=("regime",lambda s:int(s.str.startswith("BULL").sum())),
        bear_hours=("regime",lambda s:int(s.str.startswith("BEAR").sum())),
        range_hours=("regime",lambda s:int(s.str.startswith("RANGE").sum())),
        high_vol_hours=("regime",lambda s:int(s.str.endswith("HIGH_VOL").sum()))).reset_index()
    annual_market["open_close_return"]=annual_market.close_last/annual_market.open_first-1
    annual_market.to_csv(OUT/"annual_market_regimes.csv",index=False)

    bytes_before=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    raw_bytes=sum(p.stat().st_size for p in raw_files)
    candle_hashes={p.name:sha256(p) for p in candle_files}
    report={"status":"EXPLORATORY_FULL_HISTORY_INTEGRATION","generated_at_local":pd.Timestamp.now().isoformat(),
        "scope":{"execution_rows":total,"xbtusd_trade_rows":int(len(xbt)),
                 "all_execution_time_bounds_assuming_utc":[str(event_min),str(event_max)],
                 "xbtusd_trade_bounds_assuming_utc":[str(xbt_min),str(xbt_max)]},
        "execution_source_hashes":hashes,"execution_rows_by_file":per_file,"execution_source_bytes":raw_bytes,
        "hourly_market":{"source":"existing local BitMEX candle_v1 Parquet","partition_count":len(candle_files),
            "candle_count":int(len(market)),"candle_end_bounds_utc":[str(market.candle_end_utc.min()),str(market.candle_end_utc.max())],
            "missing_hours":0,"duplicates":0,"verification_artifact":"data/audit/candle_v1/candle_source_verification.json",
            "partition_hash_manifest":"hourly_partition_hashes.json"},
        "market_partition_hashes":candle_hashes,
        "integration":{"partitioning":"year=YYYY/month=MM/part-*.parquet","rows_by_join_status":dict(joins),
            "missing_timestamp_counts":dict(missing),"transacttime_timestamp_disagreements":mismatch,
            "timestamp_semantics":"Offset-free transacttime interpreted as UTC for exploratory analysis only; provenance remains UNKNOWN.",
            "asof_rule":"last candle_end_utc strictly earlier than assumed event timestamp.",
            "regime_rule":"30d return bull >= +10%, bear <= -10%, otherwise range; trailing 30d annualized hourly volatility high >=80%, otherwise normal; warmup unknown."},
        "resource":{"output_bytes_before_final_report":bytes_before,"raw_execution_bytes":raw_bytes,
            "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),"elapsed_seconds":round(time.time()-t0,2)},
        "limitations":["coverage, timestamp provenance, and position evidence remain UNKNOWN",
            "descriptive research only; no model retraining, backtest changes, or live trading",
            "only XBTUSD rows receive XBTUSD features; other symbols are retained and marked unsupported",
            "source is execution-only; unfilled order and cancellation rates cannot be established",
            "candidate position transitions are not verified account positions"]}
    write_json(OUT/"hourly_partition_hashes.json",candle_hashes)
    write_json(OUT/"research_report.json",report)
    report["resource"]["derived_bytes"]=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    write_json(OUT/"research_report.json",report)
    (OUT/"README.md").write_text(
        "# Full-history hourly execution behavior research\n\n"
        "Exploratory integration of the four original aoa execution CSVs with existing local BitMEX XBTUSD 1-hour candles. "
        "Every source row is retained; original fields remain raw strings with source lineage.\n\n"
        "Offset-free transacttime is interpreted as UTC for exploration only. Each row joins only the last hourly candle whose end is strictly earlier than the assumed event time. "
        "Regime rule: trailing 30d close return bull >= +10%, bear <= -10%, otherwise range; trailing 30d annualized volatility high >=80%, otherwise normal. "
        "Warmup regime remains unknown.\n\n"
        "Position path is reused only as an exploratory reconstruction, not verified account state. The exports are execution-only and do not support counting unfilled orders or unobserved cancellations. "
        "See research_report.json for counts, source hashes, and limitations.\n")
    print(json.dumps({"output":str(OUT),"rows":total,"xbtusd_trades":len(xbt),"candles":len(market),
        "peak_rss_bytes":report["resource"]["peak_rss_bytes"],"derived_bytes":report["resource"]["derived_bytes"],
        "joins":dict(joins),"regimes":reg.to_dict(orient="records"),"annual":annual.to_dict(orient="records"),
        "annual_market":annual_market.to_dict(orient="records"),"bootstrap":boot,"orders":order_summary,
        "candidate_positions":pos,"elapsed_seconds":report["resource"]["elapsed_seconds"]},indent=2,default=str))

if __name__=="__main__": main()
