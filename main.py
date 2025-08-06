#######################################################################
# Hedge-Fund Performance Processing Notebook | v3.1  (04-Aug-2025)
#######################################################################

# === 0 ▸ Imports & globals ====================================================
import pandas as pd, numpy as np, statsmodels.api as sm, scipy.stats as st
from math import sqrt
from pathlib import Path

DATA_DIR = Path('.')
RAW_FILE   = DATA_DIR / 'RAW DATA.xlsx'
META_FILE  = DATA_DIR / 'fund_metadata.xlsx'          # kept for future AUM etc.
OUT_FILE   = DATA_DIR / 'Processed_Results.xlsx'

MASTER_IDX = pd.date_range('2020-01-01', '2024-12-01', freq='MS')

# === 1 ▸ Load raw sheets ======================================================
xls = pd.ExcelFile(RAW_FILE)
def _clean(df): return df.loc[:, ~df.columns.str.contains('^Unnamed')]

ai_px    = _clean(xls.parse('AI Monthly_Prices 15'))
human_px = _clean(xls.parse('Human Monthly_Prices 15'))
mkt_px   = xls.parse('S&P 500')
rf_raw   = xls.parse(' Risk-free  US T-bill monthly s')
mom_fac  = xls.parse('MOM')
ff       = xls.parse('Fama-French')

# === 2 ▸ NAV interpolation → Returns =========================================
def tidy_prices(df_wide, label):
    df_wide['Date'] = pd.to_datetime(df_wide['Date'])
    frames = []
    for tkr in [c for c in df_wide.columns if c != 'Date']:
        nav = (df_wide[['Date', tkr]].rename(columns={tkr: 'NAV'})
                 .set_index('Date')
                 .reindex(MASTER_IDX))
        # drop fund if >2 consecutive missing NAVs
        gaps = nav['NAV'].isna().astype(int).groupby(nav['NAV'].notna().cumsum()).sum()
        if gaps.max() > 2:
            continue
        nav['NAV'] = (nav['NAV']
                         .interpolate(limit_area='inside')
                         .bfill()
                         .ffill())
        nav['Return'] = nav['NAV'].pct_change()
        nav = nav.dropna(subset=['Return']).reset_index().rename(columns={'index':'Date'})
        nav['Ticker'], nav['Group'] = tkr, label
        frames.append(nav)
    return pd.concat(frames, ignore_index=True)

prices = pd.concat([tidy_prices(ai_px,'AI'),
                    tidy_prices(human_px,'Human')], ignore_index=True)

# === 3 ▸ Factor matrix ========================================================
rf = (rf_raw.rename(columns={'observation_date':'Date'})
          .assign(Date=lambda d: pd.to_datetime(d['Date']),
                  RF   =lambda d: pd.to_numeric(d.iloc[:,1])/100/12)
          [['Date','RF']])

mkt_col = next(c for c in mkt_px.columns if c != 'Date')
mkt = (mkt_px.rename(columns={mkt_col:'MKT_NAV'})
              .assign(Date=lambda d: pd.to_datetime(d['Date']),
                      MKT_Return=lambda d: d['MKT_NAV'].pct_change())
              .dropna(subset=['MKT_Return'])
              [['Date','MKT_Return']])

smb = (ff.rename(columns={'DATE':'Date','Lo 30':'SMB'})
           .assign(Date=lambda d: pd.to_datetime(d['Date']),
                   SMB=lambda d: d['SMB']/100)[['Date','SMB']])
hml = (ff.rename(columns={'DATE':'Date','Hi 30':'HML'})
           .assign(Date=lambda d: pd.to_datetime(d['Date']),
                   HML=lambda d: d['HML']/100)[['Date','HML']])
mom = (mom_fac.rename(columns={'observation_date':'Date'})
                .assign(Date=lambda d: pd.to_datetime(d['Date']),
                        MOM=lambda d: d['MOM']/100)[['Date','MOM']])

factors = rf.merge(mkt,on='Date').merge(smb,on='Date')\
           .merge(hml,on='Date').merge(mom,on='Date')
factors['MKT_Excess'] = factors['MKT_Return'] - factors['RF']

# === 4 ▸ Master panel =========================================================
panel = prices.merge(factors, on='Date', how='left')
panel['Excess'] = panel['Return'] - panel['RF']
panel['Period'] = np.select(
    [
        (panel['Date']>=pd.Timestamp('2020-03-01')) &
        (panel['Date']<=pd.Timestamp('2020-04-01')),
        (panel['Date']>=pd.Timestamp('2020-05-01')) &
        (panel['Date']<=pd.Timestamp('2020-12-01')),
        (panel['Date']>=pd.Timestamp('2021-01-01'))
    ],
    ['Crash','Rebound','Post'],
    default='Full'
)

# === 5 ▸ Metric helpers =======================================================
def ann_return(r): return (1+r).prod()**(12/len(r)) - 1
def ann_vol(r):    return r.std(ddof=1)*sqrt(12)
def sharpe(ex):    return ex.mean()*12 / (ex.std(ddof=1)*sqrt(12))

def capm_alpha(excess, mkt_ex):
    if len(excess) < 12:
        return (np.nan, np.nan)
    mdl = sm.OLS(excess, sm.add_constant(mkt_ex)).fit(
              cov_type='HAC', cov_kwds={'maxlags':3})
    return mdl.params['const']*12, mdl.params[mkt_ex.name]

def ff4_alpha(ex, f):
    if len(ex) < 12: return np.nan
    mdl = sm.OLS(ex, sm.add_constant(f[['MKT_Excess','SMB','HML','MOM']])).fit(
              cov_type='HAC', cov_kwds={'maxlags':3})
    return mdl.params['const']*12

def max_dd(nav):   return (nav/nav.cummax()-1).min()

# === 6 ▸ Per-fund metrics (Full / Crash / Rebound / Post) =====================
metrics   = {}
periods   = ['Full','Crash','Rebound','Post']
for p in periods:
    sub = panel if p=='Full' else panel[panel['Period']==p]
    rows = []
    for tkr, g in sub.groupby('Ticker'):
        g = g.sort_values('Date')
        alpha_capm, beta_capm = capm_alpha(g['Excess'], g['MKT_Excess'])
        rows.append({
            'Ticker'     : tkr,
            'Group'      : g['Group'].iat[0],
            'Ann_Return' : ann_return(g['Return']),
            'Ann_Vol'    : ann_vol(g['Return']),
            'Sharpe'     : sharpe(g['Excess']),
            'CAPM_Alpha' : alpha_capm,
            'Beta'       : beta_capm,
            'FF4_Alpha'  : ff4_alpha(g['Excess'], g),
            'MaxDD'      : max_dd(g['NAV'])
        })
    metrics[p] = pd.DataFrame(rows)

# === 7 ▸ Group means, Welch-t, Mann–Whitney, Bootstrap CI =====================
cols = ['Ann_Return','Ann_Vol','Sharpe','CAPM_Alpha','FF4_Alpha','MaxDD']
gmeans, welch, mann, boot = {}, {}, {}, {}

def boot_ci(df, col, reps=5000, rng=np.random.default_rng(0)):
    ai = df[df.Group=='AI'][col].dropna().values
    hu = df[df.Group=='Human'][col].dropna().values
    if len(ai)==0 or len(hu)==0: return (np.nan, np.nan)
    diff = [rng.choice(ai, size=len(ai), replace=True).mean()
            - rng.choice(hu, size=len(hu), replace=True).mean()
            for _ in range(reps)]
    return np.percentile(diff, [2.5, 97.5])

for p, df in metrics.items():
    gmeans[p] = df.groupby('Group')[cols].mean().reset_index()

    welch[p] = pd.DataFrame([
        dict(Metric=c,
             t_stat = st.ttest_ind(df[df.Group=='AI'][c], df[df.Group=='Human'][c],
                                   equal_var=False, nan_policy='omit')[0],
             p_value= st.ttest_ind(df[df.Group=='AI'][c], df[df.Group=='Human'][c],
                                   equal_var=False, nan_policy='omit')[1])
        for c in cols])

    mann[p]  = pd.DataFrame([
        dict(Metric=c,
             U_stat = st.mannwhitneyu(df[df.Group=='AI'][c],
                                      df[df.Group=='Human'][c],
                                      alternative='two-sided')[0],
             p_value= st.mannwhitneyu(df[df.Group=='AI'][c],
                                      df[df.Group=='Human'][c],
                                      alternative='two-sided')[1])
        for c in cols])

    boot[p]  = pd.DataFrame([
        dict(Metric=c, CI_lower=boot_ci(df,c)[0], CI_upper=boot_ci(df,c)[1])
        for c in cols])

# === 8 ▸ 36-month rolling Sharpe gap (AI – Human) ============================
def rolling_sharpe(df, group, w=36):
    sub = df[df.Group==group].pivot(index='Date', columns='Ticker', values='Excess')
    rets = sub.mean(axis=1)
    roll_mean = rets.rolling(w).mean()
    roll_std  = rets.rolling(w).std()
    return (roll_mean*12) / (roll_std*sqrt(12))

gap = (rolling_sharpe(panel,'AI') - rolling_sharpe(panel,'Human')).dropna()
rolling_gap = gap.to_frame('Sharpe_Gap').reset_index()

# === 9 ▸ Write workbook =======================================================
with pd.ExcelWriter(OUT_FILE, engine='xlsxwriter') as w:
    panel.to_excel(w, 'Panel', index=False)
    for p in periods:
        metrics[p].to_excel(w, f'Metrics_{p}', index=False)
        gmeans[p].to_excel(w, f'Group_Means_{p}', index=False)
        welch[p].to_excel(w, f'Welch_T_{p}', index=False)
        mann[p].to_excel(w, f'MannWhitney_{p}', index=False)
        boot[p].to_excel(w, f'Bootstrap_CI_{p}', index=False)
    rolling_gap.to_excel(w, 'Rolling_Sharpe_Gap', index=False)

print('✔ Workbook saved to', OUT_FILE)
#######################################################################