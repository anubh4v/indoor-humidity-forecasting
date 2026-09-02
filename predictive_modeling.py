#!/usr/bin/env python3
"""
Predictive Modeling Pipeline — Forecasting indoor RH (next-day) across South Asia.

Implements the approved experimental design:
  1. Next-day prediction (daily means)                         [Decision 1]
  2. Single pooled model (household + city as features)         [Decision 2]
  3. Features: indoor RH lags (t-1,t-2,t-7), outdoor RH (t, t-1),
     monsoon flag, month, city, household                       [Decision 3]
  4. Plain lagged features, no rolling window                   [Decision 4]
  5. Baselines: persistence + linear regression                 [Decision 5]
  6. Random Forest (100 trees, fixed seed), no LSTM             [Decision 6]
  7. Single temporal 80/20 split, per-city, pooled for training [Decision 7]
  8. Evaluation per-city + pooled; MAE/RMSE/R2 + P/R/F1/acc     [Decision 8]
  9. Outputs: results table, feature importance, predicted-vs-actual,
     baseline comparison, saved model, requirements.txt         [Decision 9]

Author: Anubhav Kumar
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error,
                             r2_score, precision_score, recall_score,
                             f1_score, accuracy_score)

warnings.filterwarnings('ignore')
SEED = 42
OUTDIR = os.path.dirname(os.path.abspath(__file__))   # save outputs beside this script

# ---------------------------------------------------------------------------
# Per-city configuration (matches the corrections already encoded in Anubhav's scripts)
# ---------------------------------------------------------------------------
# city : (indoor_file_substr, encoding, rh_columns_or_None, outdoor_col)
#   rh_columns = explicit list if the author hardcoded a subset (Faisalabad, Jalna)
#                else None -> auto-detect all columns ending in "(RH)"
CITY_CONFIG = {
    'Yavatmal':    ('Yavatmal Indoor Data.csv',   'utf-8',   None,                    'Humidity'),
    'Faisalabad':  ('Faisalabad Indoor Data.csv', 'utf-8',   ['10692997 (RH)', '10692999 (RH)', '10699999 (RH)'], 'Out Hum'),
    'Dhaka':       ('Dhaka Indoor Data.csv',      'utf-8',   None,                    'Out Hum'),
    'Jalna':       ('Jalna Indoor Data.csv',      'latin-1', ['10915079 (RH) '],      'Humidity'),
    'Delhi':       ('Delhi Indoor Data.csv',      'utf-8',   None,                    'Out Hum'),
}

# Date parsing guide: the indoor files mix DD-MM-YYYY / M/D/YYYY / MM-DD-YYYY.
# pandas 'mixed' + dayfirst=False is used where unambiguous; we standardise via
# the approach the author used (regex masks then explicit formats), but fall back
# to robust parsing to cover every file.
def to_datetime_robust(df):
    """Return a Datetime Series from DD/MM/YYYY + Time columns, handling formats."""
    d = df['DD/MM/YYYY'].astype(str).str.strip()
    t = df['Time'].astype(str).str.strip()
    combined = d + ' ' + t
    # 12h AM/PM present ?
    has_ampm = combined.str.contains(r'AM|PM', regex=True, na=False).any()
    try:
        if has_ampm:
            # Normalise date separators so pandas can parse; try several formats
            out = pd.to_datetime(combined, format='mixed', dayfirst=False, errors='coerce')
        else:
            out = pd.to_datetime(combined, format='mixed', dayfirst=False, errors='coerce')
    except Exception:
        out = pd.to_datetime(combined, errors='coerce')
    # Second pass: force dayfirst for known DD-MM-YYYY-style files if first pass mostly failed
    if out.isna().mean() > 0.5 and ('-' in d.iloc[0] if len(d) else False):
        out2 = pd.to_datetime(combined, format='mixed', dayfirst=True, errors='coerce')
        out = out.where(out.notna(), out2)
    return out


def load_city(city):
    """Load indoor + outdoor for a city -> (indoor_long, outdoor_daily)."""
    cfg = CITY_CONFIG[city]
    fname, enc, rh_cols, out_col = cfg
    sdir = os.path.join(OUTDIR, city)
    indf = pd.read_csv(os.path.join(sdir, fname), encoding=enc, low_memory=False)
    # clean column names (Jalna has a trailing space, Yavatmal numbers are fine)
    indf = indf.rename(columns=lambda c: c.rstrip())

    # find RH columns
    if rh_cols is None:
        rh_cols = [c for c in indf.columns if c.strip().endswith('(RH)')]
    else:
        rh_cols = [c.rstrip() for c in rh_cols]

    # datetime
    indf['Datetime'] = to_datetime_robust(indf)
    indf = indf.dropna(subset=['Datetime'])
    indf = indf.drop_duplicates(subset=['Datetime']).set_index('Datetime')

    # daily mean indoor RH per household
    daily_parts = []
    for col in rh_cols:
        if col not in indf.columns:
            continue
        ser = indf[col].apply(pd.to_numeric, errors='coerce')
        daily = ser.resample('1D').mean().rename('indoor_rh')
        tmp = daily.to_frame()
        tmp['household'] = col
        tmp['city'] = city
        daily_parts.append(tmp)
    if not daily_parts:
        print(f"  [WARN] {city}: no RH columns usable")
        return pd.DataFrame()
    indoor_long = pd.concat(daily_parts)

    # ---- outdoor ----
    aws = os.path.join(sdir, f"{city} AWS Data.csv")
    odf = pd.read_csv(aws, encoding=enc, low_memory=False)
    odf = odf.dropna(subset=['DD/MM/YYYY'])
    odf['Datetime'] = to_datetime_robust(odf)
    odf = odf.dropna(subset=['Datetime']).drop_duplicates(subset=['Datetime']).set_index('Datetime')
    if out_col not in odf.columns:
        # fallback: first col containing 'Hum' that is outdoor-ish
        cands = [c for c in odf.columns if 'Hum' in c and 'In ' not in c]
        out_col = cands[0] if cands else None
    if out_col is None:
        print(f"  [WARN] {city}: no outdoor humidity col")
        return pd.DataFrame()
    odf[out_col] = pd.to_numeric(odf[out_col], errors='coerce')
    outdoor_daily = odf[out_col].resample('1D').mean().rename('outdoor_rh')

    return indoor_long, outdoor_daily


def build_features(indoor_long, outdoor_daily):
    """Merge indoor per-household series with outdoor, engineer lag features & target."""
    rows = []
    for (city, household), g in indoor_long.groupby(['city', 'household']):
        g = g.sort_index()
        g['indoor_rh'] = pd.to_numeric(g['indoor_rh'], errors='coerce')
        df = g[['indoor_rh']].join(outdoor_daily, how='left')
        # forward-fill outdoor RH within the household timeline to cover short gaps,
        # then fill any residual with the city-wide median (do NOT create data beyond
        # a sensible neighbourhood: ffill covers realistic continuity; median covers stragglers)
        df['outdoor_rh'] = df['outdoor_rh'].ffill().fillna(outdoor_daily.median())
        # lags
        df['lag1'] = df['indoor_rh'].shift(1)
        df['lag2'] = df['indoor_rh'].shift(2)
        df['lag7'] = df['indoor_rh'].shift(7)
        df['outdoor_lag1'] = df['outdoor_rh'].shift(1)
        # season features
        df['month'] = df.index.month
        df['monsoon'] = df.index.month.isin([6, 7, 8, 9]).astype(int)
        # target = the day itself (predict current day from prior lags + same-day outdoor)
        df['target'] = df['indoor_rh']
        df['city'] = city
        df['household'] = household
        rows.append(df)
    df = pd.concat(rows)
    df = df.dropna(subset=['lag1', 'lag2', 'lag7', 'target', 'outdoor_rh', 'outdoor_lag1'])
    return df


def mite_class(rh):
    """Map continuous RH to mite-favourable class (proxy index)."""
    if pd.isna(rh):
        return np.nan
    if rh < 50:
        return 0          # less favourable
    elif rh <= 65:
        return 1          # moderately favourable
    else:
        return 2          # highly favourable


def temporal_80_20(df):
    """Per-city temporal split: train on first 80% of unique dates, test on last 20%.
    Uses boolean masks keyed on the (possibly duplicated) Datetime index so that
    multi-household rows sharing a date are not exploded."""
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    for city in df['city'].unique():
        city_rows = (df['city'] == city)
        dates = sorted(pd.unique(df.index[city_rows]))
        if len(dates) < 5:
            train_mask |= city_rows.values
            continue
        cut = int(np.ceil(len(dates) * 0.80))
        train_dates = set(dates[:cut])
        ds = df.index[city_rows]
        train_mask[city_rows] = [d in train_dates for d in ds]
        test_mask[city_rows] = ~np.array([d in train_dates for d in ds])
    return df[train_mask], df[test_mask]


def evaluate(y_true, y_pred):
    """Regression metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }


def evaluate_class(y_true, y_pred):
    """Classification metrics on derived mite classes."""
    t = np.array([mite_class(v) for v in y_true])
    p = np.array([mite_class(v) for v in y_pred])
    return {
        'Accuracy': accuracy_score(t, p),
        'Precision(macro)': precision_score(t, p, average='macro', zero_division=0),
        'Recall(macro)': recall_score(t, p, average='macro', zero_division=0),
        'F1(macro)': f1_score(t, p, average='macro', zero_division=0),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("="*70)
    print("PREDICTIVE MODELING PIPELINE")
    print("="*70)

    all_long = []       # list of (indoor_long, outdoor_daily)
    merged = []

    for city in CITY_CONFIG:
        print(f"\n### Loading {city} ...")
        loaded = load_city(city)
        if isinstance(loaded, pd.DataFrame) and loaded.empty:
            continue
        indoor_long, outdoor_daily = loaded
        feat = build_features(indoor_long, outdoor_daily)
        if feat.empty:
            print(f"  {city}: no usable rows"); continue
        merged.append(feat)
        print(f"  {city}: {len(feat)} usable household-day rows")

    data = pd.concat(merged)
    print(f"\nTOTAL usable rows: {len(data)}")
    print("Households per city:")
    print(data.groupby('city')['household'].nunique())

    # ---- temporal split ----
    train, test = temporal_80_20(data)
    print(f"\nTrain rows: {len(train)} | Test rows: {len(test)}")

    # ---- features / target ----
    FEATURES = ['lag1', 'lag2', 'lag7', 'outdoor_rh', 'outdoor_lag1',
                'monsoon', 'month', 'city', 'household']
    X_train = train[FEATURES].copy()
    y_train = train['target'].values
    X_test = test[FEATURES].copy()
    y_test = test['target'].values

    # encode categoricals consistently
    for col in ['city', 'household']:
        cats = list(pd.unique(pd.concat([train[col], test[col]])))
        X_train[col] = pd.Categorical(X_train[col], categories=cats).codes
        X_test[col] = pd.Categorical(X_test[col], categories=cats).codes

    # ---- baselines ----
    # Persistence: predict current = last observed (lag1)
    y_persist = test['lag1'].values
    # Linear regression
    lr = LinearRegression().fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)

    # ---- overall results ----
    print("\n" + "="*70)
    print("OVERALL (pooled test) RESULTS")
    print("="*70)
    for name, yp in [('Persistence', y_persist), ('Linear', y_lr), ('RandomForest', y_rf)]:
        m = evaluate(y_test, yp)
        print(f"  {name:12s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")

    print("\nClassification (derived mite classes):")
    for name, yp in [('Persistence', y_persist), ('Linear', y_lr), ('RandomForest', y_rf)]:
        c = evaluate_class(y_test, yp)
        print(f"  {name:12s} Acc={c['Accuracy']:.3f}  P={c['Precision(macro)']:.3f}  "
              f"R={c['Recall(macro)']:.3f}  F1={c['F1(macro)']:.3f}")

    # ---- per-city results ----
    print("\n" + "="*70)
    print("PER-CITY RESULTS (Random Forest, test set)")
    print("="*70)
    rows = []
    for city in sorted(test['city'].unique()):
        msk = test['city'] == city
        gt, pr = y_test[msk], y_rf[msk]
        m = evaluate(gt, pr); c = evaluate_class(gt, pr)
        rows.append({'City': city, 'TestDays': int(msk.sum()),
                     'MAE': round(m['MAE'],3), 'RMSE': round(m['RMSE'],3), 'R2': round(m['R2'],3),
                     'Acc': round(c['Accuracy'],3), 'F1': round(c['F1(macro)'],3)})
        print(f"  {city:12s} days={msk.sum():4d}  MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"R2={m['R2']:.3f}  Acc={c['Accuracy']:.3f} F1={c['F1(macro)']:.3f}")
    percity = pd.DataFrame(rows)
    percity.to_csv(os.path.join(OUTDIR, 'results_per_city.csv'), index=False)
    print("\n  Saved: results_per_city.csv")

    # ---- feature importance ----
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 6))
    imp.plot(kind='barh', color='steelblue')
    plt.title('Random Forest Feature Importance (indoor RH next-day prediction)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print("  Saved: feature_importance.png")

    # ---- predicted vs actual (pooled or best city) ----
    best_city = percity.sort_values('R2', ascending=False).iloc[0]['City']
    msk = test['city'] == best_city
    ytrue_b = test.loc[msk.index, 'target'].sort_index()
    ypred_b = pd.Series(y_rf, index=test.index).loc[msk.index].sort_index()
    plt.figure(figsize=(12, 5))
    plt.plot(ytrue_b.index, ytrue_b.values, label='Actual', alpha=0.85)
    plt.plot(ypred_b.index, ypred_b.values, label='Predicted (RF)', alpha=0.85, linewidth=1.2)
    plt.title(f'Predicted vs Actual Indoor RH (Random Forest) — {best_city} (test period)')
    plt.xlabel('Date'); plt.ylabel('Indoor RH (%)')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'predicted_vs_actual.png'), dpi=150)
    plt.close()
    print(f"  Saved: predicted_vs_actual.png (best city = {best_city})")

    # ---- baseline comparison bar chart (R2) ----
    r2s = {'Persistence': evaluate(y_test, y_persist)['R2'],
           'Linear': evaluate(y_test, y_lr)['R2'],
           'RandomForest': evaluate(y_test, y_rf)['R2']}
    plt.figure(figsize=(8, 5))
    plt.bar(list(r2s.keys()), list(r2s.values()), color=['#999','#f4a261','#2a9d8f'])
    plt.title('Baseline Comparison — R² on pooled test set')
    plt.ylabel('R²'); plt.ylim(min(0, min(r2s.values())-0.05), 1.0)
    for i,(k,v) in enumerate(r2s.items()):
        plt.text(i, v+0.02, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'baseline_comparison.png'), dpi=150)
    plt.close()
    print("  Saved: baseline_comparison.png")

    # ---- save model + requirements ----
    import joblib
    joblib.dump(rf, os.path.join(OUTDIR, 'random_forest_model.pkl'))
    with open(os.path.join(OUTDIR, 'requirements.txt'), 'w') as f:
        f.write("pandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\nscipy\n")
    print("  Saved: random_forest_model.pkl + requirements.txt")

    # ---- overall summary table ----
    overall = pd.DataFrame({
        'Model': ['Persistence','Linear','RandomForest'],
        'MAE': [evaluate(y_test,y_persist)['MAE'], evaluate(y_test,y_lr)['MAE'], evaluate(y_test,y_rf)['MAE']],
        'RMSE':[evaluate(y_test,y_persist)['RMSE'],evaluate(y_test,y_lr)['RMSE'],evaluate(y_test,y_rf)['RMSE']],
        'R2':  [evaluate(y_test,y_persist)['R2'],  evaluate(y_test,y_lr)['R2'],  evaluate(y_test,y_rf)['R2']],
    }).round(3)
    overall.to_csv(os.path.join(OUTDIR, 'results_overall.csv'), index=False)
    print("\n  Saved: results_overall.csv")
    print("\n" + "="*70)
    print("DONE. All outputs saved to:", OUTDIR)
    print("="*70)


if __name__ == '__main__':
    main()