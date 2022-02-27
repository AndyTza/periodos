import numpy as np
from astropy.io import ascii
import os 
from tqdm import tqdm
from astropy.time import Time
import pandas as pd
import warnings
import time
import random
from gatspy import periodic, datasets
import matplotlib.pyplot as plt
from astropy.table import Table
from gatspy import datasets, periodic
import scipy.stats as sci_stat
global data_path
data_path = '../data/plasticc/data/'

# Caution, be careful when ignoring warnings!
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def generate_toi_table(data, meta_info, meta_theta_EB, meta_theta_RRL, meta_theta_AGN, meta_theta_TDE):
    """
    Generate TOI table with all the classes and light curve ID's.
    
    
    
    TODO: Need to fix documentation here and add flexibility for other models
    
    
    
    """
    id_av_rrl, id_av_eb = [], []
    id_av_agn = []
    id_av_tde = []

    for uid in tqdm(np.unique(data['object_id'])):
        ww = np.where(meta_theta_EB['object_id'] == uid)
        if np.shape(ww)[-1]==1:
            id_av_eb.append(uid)

    for uid in tqdm(np.unique(data['object_id'])):
        ww = np.where(meta_theta_RRL['object_id'] == uid)
        if np.shape(ww)[-1]==1:
            id_av_rrl.append(uid)

    for uid in tqdm(np.unique(data['object_id'])):
        ww = np.where(meta_theta_AGN['object_id'] == uid)
        if np.shape(ww)[-1]==1:
            id_av_agn.append(uid)

    for uid in tqdm(np.unique(data['object_id'])):
        ww = np.where(meta_theta_TDE['object_id'] == uid)
        if np.shape(ww)[-1]==1:
            id_av_tde.append(uid)

    id_av_rrl, id_av_eb, id_av_agn = np.array(id_av_rrl), np.array(id_av_eb), np.array(id_av_agn)
    id_av_tde = np.array(id_av_tde)

    _id1 = np.array(['rrl' for _ in range(len(id_av_rrl))])
    _id2 = np.array(['eb' for _ in range(len(id_av_eb))])
    _id3 = np.array(['agn' for _ in range(len(id_av_agn))])
    _id4 = np.array(['tde' for _ in range(len(id_av_tde))])



    # All ID's and & ID tags
    all_id = np.concatenate([id_av_rrl, id_av_eb, id_av_agn, id_av_tde])
    _id_all = np.concatenate([_id1, _id2, _id3, _id4])

    # Final TOI table
    toi_table = Table([all_id, _id_all], names=('obj_id', 'type'))

    return toi_table





# Read ALL PlastiCC data & meta
data = pd.read_csv(data_path + "plasticc_train_lightcurves.csv.gz",
                  compression='gzip',
                  error_bad_lines=False)

meta_info = ascii.read(data_path + "plasticc_train_metadata.csv") # ascii meta since it's smaller
meta_theta_EB = ascii.read(data_path + 'plasticc_modelpar/' + 'plasticc_modelpar_016_EB.csv')
meta_theta_RRL = ascii.read(data_path + 'plasticc_modelpar/' + 'plasticc_modelpar_092_RRL.csv')
meta_theta_AGN = ascii.read(data_path + "plasticc_modelpar/" + "plasticc_modelpar_088_AGN.csv")
meta_theta_TDE = ascii.read(data_path + "plasticc_modelpar/" +"plasticc_modelpar_015_TDE.csv")

# Fetch all toi's
toi_table = generate_toi_table(data, meta_info, meta_theta_EB, meta_theta_RRL, meta_theta_AGN, meta_theta_TDE)

# Helper functions to read and digest plasticc data

def generate_lc(obj_id, band='all', data_table=data, det=1):
    """Unpack and return PlastiCC data in numpy array format.
    
    Input:
    ------
    obj_id: Object ID
    band: Photometric bandpass filter. 'all' includes ugrizy, or 'ugrizy'
    data_table: Pandas data table containing the light curves
    det: Detection from the image subtraction algorithm. ==1 detection, ==0 not detection (i.e upper limit)
    """
    
    data_table_mod = data_table[data_table['detected_bool']==det]
    # Select light curve based on the ID 
    lc = data_table_mod[data_table_mod['object_id']==obj_id]
    
    if len(lc)==0:
        print ("Sorry, it seems like your obj_id query was wrong!")
        return False
    
    lsst_bands = list('ugrizy') # lsst photomeric bands
    
    lc_array = lc.to_numpy()
    mjd, flux, flux_err = lc_array[:,1], lc_array[:,3], lc_array[:,4]
    flt = lc_array[:,2].astype(int).astype(str)    
    
    for j in range(6):
        flt[flt==str(j)] = lsst_bands[j]
    
    if band=='all':
        return mjd, flux, flux_err, flt
    else:
        return mjd[flt==band], flux[flt==band], flux_err[flt==band], flt[flt==band]
     

def fetch_type(lid, table=toi_table):
    return table[table['obj_id']==lid]


def fetch_meta_info(lc_id, lc_type):
    if lc_type=='rrl':
        # crossmatch to approprirate table
        xm_ = np.where(meta_theta_RRL['object_id']==lc_id)
        return meta_theta_RRL[xm_]
    elif lc_type=='eb':
        # crossmatch to approprirate table
        xm_ = np.where(meta_theta_EB['object_id']==lc_id)
        return meta_theta_EB[xm_]
    elif lc_type=='agn':
        # crossmatch to approprirate table
        xm_ = np.where(meta_theta_AGN['object_id']==lc_id)
        return meta_theta_AGN[xm_]

# Write a function that will generate N random from each class (equal)
def draw_rand_trans(table, N=10, class_type='rrl'):
    """Given N this function will draw an equal number of trnasinets.
       Note: It will not draw the same transiennt
    """
    # isolate each unique class
    req_tab = table[table['type']==class_type]  
    
    # Random number generator w/o repeat
    rng = np.random.default_rng()
    rn = rng.choice(len(req_tab), size=N, replace=False)
    
    return req_tab[rn]  

def run_multi_lsp(x, y, err, fts, fmin=0.1, fmax=150, k=1, mode='fast', dt_cut=365):
    """Run the mutliband LSP from gatspy. Supports fast or generalized"""
    
    try:
        # Pre-processing to photometry
        dt = x-x[0] # calculate transient duration
        x, y, err, fts = x[dt<=365], y[dt<=365], err[dt<=365], fts[dt<=365]
        y += -1*min(y)
        dt = x-x[0]
        # Check fmax limit
        if max(dt)<fmax:
            fmax = max(dt)-5 
    except:
        return np.nan
    
    if mode=='fast':
        try:
            model = periodic.LombScargleMultibandFast(fit_period=True,optimizer_kwds={"quiet": True},
                                  Nterms=k)
            model.optimizer.set(period_range=(fmin, fmax))
            model = model.fit(x, y, dy=err, filts=fts)
            return model.best_period
        except:
            return np.nan
    elif mode=='general':
        try:
            model = periodic.LombScargleMultiband(fit_period=True,optimizer_kwds={"quiet": True},
                      Nterms_base=k, Nterms_band=k)
            model.optimizer.set(period_range=(fmin, fmax))
            model = model.fit(x, y, dy=err, filts=fts)
            return model.best_period
        except:
            return np.nan
        

def run_single_lsp(x, y, err, fts, band='u', fmin=0.1, fmax=150, k=1, mode='fast', dt_cut=365):
    """Run the mutliband LSP from gatspy. Supports fast or generalized"""
    
    try:
        # Pre-processing to photometry
        dt = x-x[0] # calculate transient duration
        x, y, err, fts = x[dt<=365], y[dt<=365], err[dt<=365], fts[dt<=365]
        y += -1*min(y)
        dt = x-x[0] # updated dt
        
        # isolate photometric band
        x, y, err = x[fts==band], y[fts==band], err[fts==band]
        
        # Check fmax limit
        if max(dt)<fmax:
            fmax = max(dt)-5 
        
    except:
        return np.nan
    
    if mode=='fast':
        try:
            model = periodic.LombScargleFast(fit_period=True,optimizer_kwds={"quiet": True},
                                  Nterms=1)
            model.optimizer.set(period_range=(fmin, fmax))
            model = model.fit(x, y, dy=err)
            return model.best_period
        except:
            return np.nan
    elif mode=='general':
        try:
            model = periodic.LombScargleMultiband(fit_period=True,optimizer_kwds={"quiet": True},
                      Nterms_base=k)
            model.optimizer.set(period_range=(fmin, fmax))
            model = model.fit(x, y, dy=err)
            return model.best_period
        except:
            return np.nan
  

def generate_tags(kmax):
    """Generate titles for master table on LSP analysis"""
    # Create data table
    m_lsp_name_fast_list = []
    m_lsp_name_gen_list = []
    for i in range(kmax):
        m_lsp_name_fast_list.append('multi_lsp_f'+f'{i+1}')
        m_lsp_name_gen_list.append('multi_lsp_g'+f'{i+1}')
        
    s_lsp_gen_list = []
    for iii in range(kmax):
        for jj, band_name in enumerate(list('ugrizy')):
            s_lsp_gen_list.append('s_lsp_g'+f'{iii+1}'+f'_{band_name}')
            
    s_lsp_fast_list = [] 
    for band_name in list('ugrizy'):
        s_lsp_fast_list.append(f's_lsp_f_{band_name}')
        
    master_names = np.concatenate([['id'], ['ndet'], ['ptrue'], m_lsp_name_fast_list, m_lsp_name_gen_list, s_lsp_gen_list, s_lsp_fast_list])
    
    return master_names

def calc_all_lsp(N, transient_class='rrl', k_max_comp=7, table=toi_table):
    """DOCS
    
    Will return a master table (ascii.Table) of both General & Fast LSP (single & multi)
    
    Output:
    -------
    Table content will contain the following parameters:
    lc_id: Light curve ID
    N_det: Number of photometric detections (multi: ugrizy; single: band)
    multi_f1...multi_fk: Multiband fast LSP (kth component)
    multi_g1...multi_gk: Multibamd general LSP (kth component)
    ugrizy_g1...urizy_gk: Single general LSP (kth component)
    urizy_f1: Single fast LSP (k=1)
    p_true: True injected period (0 if not-periodic variable)
    """
    
    # Generalte N class
    toi_special = draw_rand_trans(table,
                                  N=N,
                                  class_type=transient_class)
    _id_unq = toi_special['obj_id'].data # all the ID's of the unique class
    ndet = np.zeros(shape=N)
    ptrue = np.zeros(shape=N)
    M_multi_fast = np.zeros(shape=(N,k_max_comp))
    M_multi_general = np.zeros(shape=(N,k_max_comp))
    M_single_fast = np.zeros(shape=(N, 6)) # ugrizy columns
    M_single_general = np.zeros(shape=(N, k_max_comp, 6))
    
    i = 0 
    for _id in tqdm(_id_unq):
        
        # Fetch photometry
        t, mag, mag_err, flt = generate_lc(_id)
        
        for j in range(k_max_comp):
            # multi-LSP
            m_f_lsp = run_multi_lsp(t, mag, mag_err, flt, k=j, mode='fast')
            m_g_lsp = run_multi_lsp(t, mag, mag_err, flt, k=j, mode='general')
            
            for ii, flt_lst in enumerate(list('ugrizy')):
                # Single-band lsp per band
                X_g_lsp = run_single_lsp(t, mag, mag_err, flt, band=flt_lst, k=j, mode='general') # general
                M_single_general[i, j, ii] = X_g_lsp # append data
                
            M_multi_fast[i, j]=m_f_lsp
            M_multi_general[i, j]=m_g_lsp
        
        # fast single_lsp
        for ii, flt_lst in enumerate(list('ugrizy')):
            X_f_lsp = run_single_lsp(t, mag, mag_err, flt, band=flt_lst, mode='fast') # general
            M_single_fast[i, ii] = X_f_lsp

        if transient_class=='rrl' or transient_class=='eb':
            f_tp = fetch_meta_info(_id, lc_type=transient_class)['PERIOD'].data[0]
        else:
            f_tp = np.nan
        
        ptrue[i] = f_tp
        ndet[i] = len(t)
           
        i+=1
        
    master_names = generate_tags(k_max_comp) # generate names for table  
    Table_master = Table(names=master_names)
    #master_names = np.concatenate([['id'], ['ndet'], ['ptrue'], m_lsp_name_fast_list, m_lsp_name_gen_list, s_lsp_gen_list, s_lsp_fast_list])

    for i in range(N):
        # Collapse multi matrix
        M_LSP_F = M_multi_fast[i]
        M_LSP_G = M_multi_general[i]
        
        # Collapse single matrix
        S_LSP_U = M_single_general[i, :, 0]
        S_LSP_G = M_single_general[i, :, 1]
        S_LSP_R = M_single_general[i, :, 2]
        S_LSP_I = M_single_general[i, :, 3]
        S_LSP_Z = M_single_general[i, :, 4]
        S_LSP_Y = M_single_general[i, :, 5]
        
        S_LSP_fast = M_single_fast[i]
        
        master_var_col = np.concatenate([[_id_unq[i]], [ndet[i]], [ptrue[i]], 
                                        M_LSP_F, M_LSP_G,
                                         S_LSP_U, S_LSP_G, S_LSP_R, S_LSP_I, S_LSP_Z, S_LSP_Y,
                                        S_LSP_fast])
        
        Table_master.add_row(master_var_col)
        
    
    
    # Store table!
    Table_master.write(f"../data/{transient_class}_master_N{N}", format='ascii')
    return Table_master
            

def main():
    calc_all_lsp(100, transient_class='rrl', k_max_comp=7)

if __name__ == "__main__":
    main()
