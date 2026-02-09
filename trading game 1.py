#!/usr/bin/env python3

import datetime as dt
import warnings
import sys
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

# Vérification des bibliothèques nécessaires
try:
    import gurobipy as gp
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
    from statsmodels.tsa.stattools import adfuller
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print("Assurez-vous d'avoir installé : gurobipy, matplotlib, numpy, pandas, scipy, statsmodels")
    sys.exit(1)

# Configuration Matplotlib (Utilisation du backend par défaut pour éviter les erreurs Qt5)
# matplotlib.use('Qt5Agg', force=True) 
plt.style.use('seaborn-v0_8-deep') # Mise à jour pour les versions récentes de mpl

# --- CONSTANTES ---
CROSSING_MEAN = 2.0  # Ajusté pour les actions (souvent moins volatiles que la crypto minute par minute)
CROSSING_MAX = 4.0
ORIG_AMOUNT = 10000
SAVE_FILE = None
RISK_FREE_RATE = 4/(100)      # 4-week T-bill return rate
TX_COST = 0.001
LAMBDA = 0.5

# --- FONCTIONS D'OPTIMISATION & TRADING (Code Original Adapté) ---

def build_prob_cons(longs, shorts, prices, expected_returns,
                     expected_risk, trading_weights, risk_vec,
                    index, LAMBDA, tc):
    
    global CROSSING_MAX, CROSSING_MEAN

    # XXX: Make the model
    try:
        model = gp.Model('portfolio')
    except gp.GurobiError as e:
        print(f"Erreur Gurobi: {e}. Avez-vous une licence valide ?")
        return None, []

    # XXX: The weight vars to optimise
    lws = [model.addVar(name='l!%s!%s' % (v[0], v[1]), lb=0, ub=1)
           for v in longs]
    sws = [model.addVar(name='s!%s!%s' % (v[0], v[1]), lb=-1, ub=0)
           for v in shorts]
    model.update()

    # XXX: Get the expected returns
    return_vec = [None]*len(shorts)
    for i, v in enumerate(shorts):
        if v[1] == v[0].split('_')[1]:
            return_vec[i] = -expected_returns[v[0]][0]
        else:
            return_vec[i] = -expected_returns[v[0]][1]

    # XXX: Return optimisation objective
    max_ret_s = np.array(sws).dot(np.array(return_vec).T)

    return_vec = [None]*len(longs)
    for i, v in enumerate(longs):
        if v[1] == v[0].split('_')[1]:
            return_vec[i] = expected_returns[v[0]][0]
        else:
            return_vec[i] = expected_returns[v[0]][1]

    max_ret_l = np.array(lws).dot(np.array(return_vec).T)
    max_ret = max_ret_s + max_ret_l

    # XXX: Risk part of the objective
    ws = [None]*len(risk_vec)
    for j, (f, s) in enumerate(zip(sws, lws)):
        parts_f = f.VarName.split('!')
        parts_s = s.VarName.split('!')
        # On s'assure qu'on compare bien les mêmes paires
        if parts_f[2] < parts_s[2]:
            ws[j] = np.array((f, s))
        else:
            ws[j] = np.array((s, f))
    
    # XXX: Covariance matrix adjustments
    for k in risk_vec:
        risk_vec[k].iat[0, 1] = -risk_vec[k].iat[0, 1]
        risk_vec[k].iat[1, 0] = -risk_vec[k].iat[1, 0]

    min_risk = np.sum([(ws[i].dot(r).dot(ws[i].T))
                       for i, r in enumerate(risk_vec.values())])

    # XXX: Set Objective
    obj = max_ret - (LAMBDA*min_risk)
    model.setObjective(obj, gp.GRB.MAXIMIZE)
    model.update()

    # XXX: Overall cash available constraint (CORRECTION ICI)
    currs = [ll.VarName.split('!')[2] for ll in lws]
    currs += [ss.VarName.split('!')[2] for ss in sws]
    currs = set(currs)
    
    c_cons = [None]*len(currs)
    for k, c in enumerate(currs):
        if tc:
            # Calcul des expressions mathématiques (LinExpr)
            lws_tc = [l*(1+TX_COST) for l in lws]
            sws_tc = [s*(1-TX_COST) for s in sws]
            
            # FILTRAGE CORRIGÉ :
            # On itère simultanément sur la variable (l/s) pour vérifier le nom (.VarName)
            # et sur l'expression (expr) pour l'ajouter à la somme.
            ws_long = [expr for l, expr in zip(lws, lws_tc) if l.VarName.split('!')[2] == c]
            ws_short = [expr for s, expr in zip(sws, sws_tc) if s.VarName.split('!')[2] == c]
            
            ws = ws_long + ws_short
        else:
            ws = [w for w in (lws + sws) if w.VarName.split('!')[2] == c]
            
        c_cons[k] = (1-np.sum(trading_weights[c] + ws) >= 0)
        model.addConstr(c_cons[k])
    
    # XXX: Price constraint
    p_cons = [None]*len(lws)
    for j, (ll, ss) in enumerate(zip(lws, sws)):
        lindex = int(ll.VarName.split('!')[2])
        sindex = int(ss.VarName.split('!')[2])
        
        # Ici ça fonctionne car ll et ss sont les variables originales
        if tc:
            p_cons[j] = (-1*ss*((prices[lindex][index]*(1+TX_COST)))/(prices[sindex][index]*(1-TX_COST))== ll)
        else:
            p_cons[j] = (-1*ss*(prices[lindex][index]/prices[sindex][index])== ll)
        model.addConstr(p_cons[j])

    return model, zip(lws, sws)

def simulate_trade(spreads, sigma3, lookup, spread_dates,
                   expected_returns, expected_risk, prices_df_list,
                   start, ed, zscores, tc):
    
    global LAMBDA

    # XXX: The required prices - adaptation: prices_df_list is a list of DFs (one per ticker)
    # We extract numpy arrays for the simulation
    prices = []
    for p_df in prices_df_list:
        # p_df has 'Date' column with timestamps and 'Close'
        mask = p_df['Date'].isin(spread_dates)
        prices.append(p_df.loc[mask, 'Close'].to_numpy())

    max_p_index = 0 if start is None else start
    end_p_index = spread_dates.shape[0] if ed is None else ed
    
    # XXX: Dates conversion from timestamp
    start_d = dt.datetime.fromtimestamp(spread_dates[max_p_index])
    end_d = dt.datetime.fromtimestamp(spread_dates[end_p_index-1])
    start_date = start_d.strftime('%Y-%m-%d')
    end_date = end_d.strftime('%Y-%m-%d')
    
    print('START DATE: ', start_date)
    print('END DATE: ', end_date)
    trading_days = (end_d - start_d).days
    trading_days = 1 if trading_days <= 0 else trading_days
    print('# of days: ', trading_days)

    trading_weights = {k: [] for k in lookup.keys()}
    trading_indices = {k: [max_p_index] for k in lookup.keys()}

    # XXX: Only one open position/spread at any given time
    open_positions = {k: None for k in spreads.columns}
    orig_amount = {k: ORIG_AMOUNT for k in lookup.keys()}

    TOTAL = {k: 0 for k in spreads.columns}

    # Simulation Loop
    for i in range(max_p_index+1, end_p_index):
        
        # XXX: Close any open position
        for k in open_positions.keys():
            dev = spreads[k].std(axis=0)
            if ((open_positions[k] is not None) and abs(zscores[k][i]) < CROSSING_MEAN*dev):
                TOTAL[k] += 1     
                sindex = open_positions[k][1]
                lindex = open_positions[k][2]
                Eth = open_positions[k][0] # "Eth" here just means "Base Amount" unit
                
                ps = prices[int(sindex)][i]*(1-TX_COST if tc else 1)
                pl = prices[int(lindex)][i]*(1+TX_COST if tc else 1)
                
                ws = -1*(Eth*ps)/orig_amount[sindex]
                wl = (Eth*pl)/orig_amount[lindex]
                
                trading_weights[sindex].append(ws)
                trading_weights[lindex].append(wl)
                
                trading_indices[sindex].append(i)
                trading_indices[lindex].append(i)

                open_positions[k] = None

        # XXX: Get the longs and the shorts
        longs = list()
        shorts = list()
        risk_vec = dict()
        
        for s in spreads.columns:
            if (abs(zscores[s][i]) > sigma3[s] and (open_positions[s] is None)):
                risk_vec[s] = expected_risk[s]
                skeys = s.split('_')
                # skeys structure: s_Index1_Index2
                idx1 = skeys[1]
                idx2 = skeys[2]
                
                if zscores[s][i] > 0:
                    shorts.append((s, idx1))
                    longs.append((s, idx2))
                else:
                    shorts.append((s, idx2))
                    longs.append((s, idx1))

        assert(len(longs) == len(shorts))
        if (len(longs) > 0):
            problem, lsws = build_prob_cons(longs, shorts,
                                            prices, expected_returns,
                                            expected_risk,
                                            trading_weights, risk_vec, i,
                                            LAMBDA, tc)
            if problem is None: continue # Skip if Gurobi failed
            
            problem.Params.OutputFlag = 0
            problem.optimize()

            if problem.status == gp.GRB.INFEASIBLE:
                continue

            # XXX: Open trade positions and weights
            for ll, s in lsws:
                lindex = ll.VarName.split('!')[2]
                sindex = s.VarName.split('!')[2]
                
                ps = prices[int(sindex)][i]*(1-TX_COST if tc else 1)
                pl = prices[int(lindex)][i]*(1+TX_COST if tc else 1)
                
                # Protect against zero division if price is 0 (unlikely in stocks but possible in data errors)
                if pl == 0 or ps == 0: continue

                Ethl = (orig_amount[lindex]*ll.X)/pl   # Amount to buy
                Eths = (orig_amount[lindex]*s.X)/ps    # Amount to sell
                
                Ethl = float('%0.6f' % Ethl)
                Eths = float('%0.6f' % Eths)
                
                # Check consistency
                # assert Ethl == abs(Eths), ('%f != %f' % (Ethl, abs(Eths)))

                trading_weights[lindex].append(ll.X)
                trading_weights[sindex].append(s.X)

                assert(ll.VarName.split('!')[1] == s.VarName.split('!')[1])
                open_positions[ll.VarName.split('!')[1]] = (Ethl, lindex, sindex)

                trading_indices[sindex].append(i)
                trading_indices[lindex].append(i)

    # Clean up open positions at the end
    for k in open_positions.keys():
        if open_positions[k] is not None:
            lkey = open_positions[k][1]
            skey = open_positions[k][2]
            trading_weights[lkey] = trading_weights[lkey][:-1]
            trading_weights[skey] = trading_weights[skey][:-1]
            trading_indices[lkey] = trading_indices[lkey][:-1]
            trading_indices[skey] = trading_indices[skey][:-1]
            open_positions[k] = None

    # XXX: Visualization Part
    # We only visualize the first 4 assets to avoid cluttering if universe is large
    keys_to_plot = list(lookup.keys())[:4]
    
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
    
    df_vals = pd.DataFrame()
    for k in keys_to_plot:
        # Ensure parity in trades
        if len(trading_weights[k]) % 2 != 0:
             trading_weights[k] = trading_weights[k][:-1]
             
        pl_val = (1-np.sum(trading_weights[k]))*orig_amount[k]-orig_amount[k]
        print('Profit/Loss %s: %f' % (lookup[k], pl_val))
        
        buyeth = orig_amount[k]/prices[int(k)][max_p_index] * (1-TX_COST if tc else 1)
        
        toplot = (prices[int(k)][max_p_index:end_p_index])*buyeth/orig_amount[k]
        xlabels = [start_date, end_date]
        xticks = [0, len(toplot)-1]

        ax[0].plot(toplot, label=lookup[k])
        ax[0].set_xticks(xticks, xlabels)
        ax[0].set_ylabel('Price Index (Normalized)')
        ax[0].legend()

        # Profit curve
        pl_curve = pd.Series(trading_weights[k]).cumsum()
        ones = np.ones(len(pl_curve))
        val = (ones-pl_curve)*orig_amount[k]
        val = list(val)
        val.insert(0, orig_amount[k])
        
        # Mapping values to dates
        t_ds = pd.DataFrame(columns=['Date', ('P/L_%s' % lookup[k])])
        t_ds['Date'] = spread_dates
        
        current_indices = trading_indices[k][:len(val)] # Safety slice
        for j, idx in enumerate(current_indices):
            t_ds.loc[idx, ('P/L_%s' % lookup[k])] = val[j]
            
        val_filled = t_ds.fillna(method='ffill')
        val_final = list(val_filled[('P/L_%s' % lookup[k])][max_p_index:end_p_index])

        # Calmar Ratio Calc
        pp = pd.Series(val_final)
        avgpp = (pp.iloc[-1]-orig_amount[k])/orig_amount[k]
        ann_avgpp = ((1 + avgpp)**(365/trading_days)) - 1
        
        cum_max = np.maximum.accumulate(pp)
        drawdown_series = (cum_max - pp) / cum_max
        max_drawdown = drawdown_series.max()
        
        if max_drawdown == 0: max_drawdown = 1 # Avoid div by zero
        
        calmar_ratio_t = (ann_avgpp-RISK_FREE_RATE)/max_drawdown
        print('Our Calmar ratio %s: %f, cum return: %f%%, annual return: %f%%, drawdown: %f%%' %
              (lookup[k], calmar_ratio_t, (avgpp*100), (ann_avgpp*100), max_drawdown*100))

        ax[1].plot(val_final, label=lookup[k])
        ax[1].set_ylabel('Portfolio Value')
        ax[1].set_xticks(xticks, xlabels)
        ax[1].legend()

    plt.tight_layout()
    plt.show(block=True)

def keytoCUR(key, lookup):
    keys = key.split('_')
    return '_'.join([keys[0], lookup[keys[1]], lookup[keys[2]]] + keys)

def get_expected_returns(dfs):
    # First compute the log returns for the dfs
    # dfs is a list of dataframes with 'Close' column
    rets = [df['Close'].transform(np.log).diff().dropna()
            for df in dfs]
    return rets

def pairwise_cov_matrices(dfs):
    # XXX: First get the log returns for each day
    rets = [df['Close'].transform(np.log).diff() for df in dfs]
    rets = [pd.DataFrame({'Date': d['Date'], 'Close': r})
            for d, r in zip(dfs, rets)]
    rets = [r.dropna() for r in rets]

    # XXX: Now do pairwise convariane matrices
    covdf = dict()
    for i in range(len(rets)):
        idates = set(rets[i]['Date'])
        for j in range(i+1, len(rets)):
            cdf = pd.DataFrame()
            jdates = set(rets[j]['Date'])
            cdates = jdates.intersection(idates)

            cdfi = rets[i][rets[i].Date.isin(cdates)]
            cdfj = rets[j][rets[j].Date.isin(cdates)]
            
            # Align by sorting by date just in case
            cdfi = cdfi.sort_values('Date')
            cdfj = cdfj.sort_values('Date')

            cdf['Close_%d' % i] = cdfi['Close'].to_numpy()
            cdf['Close_%d' % j] = cdfj['Close'].to_numpy()

            covdf['s_%d_%d' % (j, i)] = cdf.cov()
    return covdf

def get_expected_time_to_mean(oc_dates):
    diff = {k: np.mean([vv[1]-vv[0] for vv in v]) for k, v in oc_dates.items()}
    return pd.Series(diff)

def get_close_date(start, spread, spreads, spread_dates, zscores):
    dev = spreads[spread].std(axis=0)
    toret = spread_dates[start]
    index = spreads.shape[0]
    
    for i in range(start, spreads.shape[0]):
        if(abs(zscores[spread][i]) < CROSSING_MEAN*dev):
            toret = spread_dates[i]
            index = i
            break
    return index, toret

def sigma3(dfs_list, spreads, lookup, spread_dates, start, end, zscores, tc):
    # dfs_list: list of dataframes (one per ticker)
    
    ret_vec = get_expected_returns(dfs_list)
    ret_vec = [v.mean() for v in ret_vec]
    
    stds = spreads.std(axis=0)
    sigma3_val = stds*CROSSING_MAX

    open_close_dates = {s: list() for s in spreads.columns}
    c_index = {s: -1 for s in spreads.columns}
    
    for i in range(spreads.shape[0]):
        for s in spreads.columns:
            if((abs(zscores[s][i]) > sigma3_val[s]) and (i > c_index[s])):
                ci, cdate = get_close_date(i, s, spreads, spread_dates, zscores)
                c_index[s] = ci
                if cdate > spread_dates[i]:
                    open_close_dates[s].append((spread_dates[i], cdate))

    # Clean empty lists to avoid nan in mean calculation
    open_close_dates = {k:v for k,v in open_close_dates.items() if len(v) > 0}
    
    if len(open_close_dates) == 0:
        print("No mean reversion opportunities found with current parameters.")
        return

    time_to_mean_vec = get_expected_time_to_mean(open_close_dates)
    cov_matrices = pairwise_cov_matrices(dfs_list)

    expected_returns = pd.DataFrame()
    for s in time_to_mean_vec.index:
        keys = s.split('_') # s_Index1_Index2
        idx1 = int(keys[1])
        idx2 = int(keys[2])
        
        term1 = ret_vec[idx1]*time_to_mean_vec[s]
        term2 = ret_vec[idx2]*time_to_mean_vec[s]
        
        if not tc:
            expected_returns[s] = [term1, term2]
        else:
            expected_returns[s] = [
                term1 - TX_COST*abs(term1), 
                term2 - TX_COST*abs(term2)
            ]

    expected_risk = dict()
    for s in time_to_mean_vec.index:
        if s in cov_matrices:
            expected_risk[s] = cov_matrices[s]*time_to_mean_vec[s]
    
    # Filter spreads to only those we have risk/return data for
    valid_spreads = spreads[expected_returns.columns]
    
    simulate_trade(valid_spreads, sigma3_val, lookup, spread_dates,
                   expected_returns, expected_risk, dfs_list, start, end,
                   zscores, tc)

# --- CORE LOGIC FOR COPPER DATA ---

def run_copper_strategy():
    filename = "copper_project_data.csv"
    if not os.path.exists(filename):
        print(f"Fichier {filename} introuvable. Veuillez d'abord exécuter le script de téléchargement.")
        return

    print("Chargement des données Copper Project...")
    # Load and prep data
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    df = df.ffill().dropna() # Ensure clean data
    
    # Create the lookup dictionary dynamically (0: TickerA, 1: TickerB...)
    tickers = df.columns.tolist()
    lookup = {str(i): t for i, t in enumerate(tickers)}
    print(f"Univers ({len(tickers)} actifs) : {tickers}")

    # --- Prepare Data Structures for the Legacy Logic ---
    # The old logic expects:
    # 1. 'dfs': A combined dataframe with columns named Close_0, Close_1...
    # 2. 'odfs': A list of individual dataframes with 'Date' and 'Close'
    # 3. 'spread_dates': A numpy array of UNIX timestamps
    
    # Convert Index to Unix Timestamp (float) for compatibility
    df['Timestamp'] = df.index.astype('int64') // 10**9
    spread_dates = df['Timestamp'].to_numpy()
    
    # Create 'dfs' (Combined for regression)
    dfs_combined = pd.DataFrame()
    for i, ticker in enumerate(tickers):
        dfs_combined[f'Close_{i}'] = df[ticker].values

    # Create 'odfs' (List for passing to functions)
    odfs = []
    for i, ticker in enumerate(tickers):
        temp_df = pd.DataFrame()
        temp_df['Date'] = df['Timestamp'].values # Use timestamp for logic
        temp_df['Close'] = df[ticker].values
        odfs.append(temp_df)

    # --- Compute Spreads & Z-Scores ---
    print("Calcul des spreads et des corrélations...")
    dfs_log = dfs_combined.transform(np.log)
    spreads = pd.DataFrame()
    cols = dfs_log.columns
    
    # Limit number of pairs if universe is too big (20 assets = 190 pairs, OK)
    pair_count = 0
    for i in range(dfs_log.shape[1]):
        for j in range(i+1, dfs_log.shape[1]):
            res = linregress(dfs_log[cols[i]], dfs_log[cols[j]])
            # Residuals = Spread
            X = (dfs_log[cols[j]] - ((dfs_log[cols[i]]*res.slope) + res.intercept))
            
            # ADF Test (Optional for speed, enabled here)
            _, pvalue, _, _, _, _ = adfuller(X, regression='ct')
            if pvalue < 0.1: # Only trade stationary spreads
                spreads['s_%d_%d' % (j, i)] = X
                pair_count += 1
    
    print(f"Paires cointégrées identifiées : {pair_count}")
    
    if pair_count == 0:
        print("Aucune paire cointégrée trouvée. Fin.")
        return

    # Calculate Z-Scores
    zscores = pd.DataFrame(columns=spreads.columns)
    for k in spreads.columns:
        zscores[k] = (spreads[k]-spreads[k].mean())/spreads[k].std(ddof=0)

    # --- Run Strategy ---
    # Parameters: start index, end index
    # We use roughly the last 70% of data for trading simulation
    start_idx = int(len(spread_dates) * 0.3)
    end_idx = len(spread_dates)
    
    print("Démarrage de la simulation de trading...")
    sigma3(odfs, spreads, lookup, spread_dates, start_idx, end_idx, zscores, tc=True)


def main():
    run_copper_strategy()

if __name__ == '__main__':
    main()