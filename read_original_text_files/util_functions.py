#!/usr/bin/env python
# coding: utf-8



#import streamlit as st

import pandas as pd
from datetime import date
from datetime import datetime
import time 
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib
import seaborn as sns
import numpy as np

from IPython.display import display


def setPandasOptions():
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    pd.set_option('display.max_seq_items', 100)
    pd.set_option('display.precision', 2)


# ### Utility functions


def upperToLower(df):
    # Create dictionaries from upper to lower and lower to upper column names
    upper = df.columns
    u2l = {}
    l2u = {}
    for col in upper:
        low = col.lower()
        u2l[col] = low
        l2u[low] = col
    return u2l, l2u


def getSortedHourCount(series, msg="", nrows=10):
    print(msg)
    fsudate = pd.to_datetime(series).astype(int)
    fsutime = pd.to_datetime(fsudate).dt.strftime("%H")
    return fsutime.str[0:2].value_counts().sort_index().iloc[0:nrows]


def new_cleanupColumns(df1):
    """
    Cleanup the columns of the FSU tables provided by Copa in Jan. 2021.

    Parameters: 
        df1: DataFrame to clean

    Return: a tuple with two objects: 
       df: the cleaned dataframe
       non_convertad: a list of column names that are not strings and could not be converted to 'float' or 'int'
       

    Parameters: DataFrame to clean
    No columns are deleted. Character columns are stripped (blank spaces removed in front and back). 
    Some columns are converted to int. 
    Time columns (DTMZ) only keep hours and minutes. 
    Date columns are changed so that the dates are sortable (yyyy/mm/dd) as strings. 

    Return: modified dataframe.
    """
    df = df1.copy()
    non_converted = []

    # Columns with special treatment
    #df['ARR_DELAY_MINUTES'] = df['ARR_DELAY_MINUTES'].replace('?', 999999).astype(float).astype(int)
    #df['DEP_DELAY_MINUTES'] = df['DEP_DELAY_MINUTES'].replace('?', 999999).astype(float).astype(int)

    # Replace all other '?' by NaN
    df.replace('?', np.nan, inplace=True)
        
    # Strip 'object' or 'str' types
    # Convert other columns to integers after converting NaN
    for col in df.columns:
        if df[col].dtype == 'O' or df[col].dtype == 'str':
            df[col] = df[col].str.strip()
        else:
            try:
                pass
                df[col] = df[col].astype('int')
            except:
                try:
                    df[col] = df[col].fillna(-1).astype('int')
                    df[col] = df[col].astype('int')
                except:
                    non_converted.append(col)
    # string format
    #df.SCH_DEP_TMZ = df.SCH_DEP_TMZ.str[0:5] # not correct
    # datetime format
    #df.SCH_DEP_DTZ = pd.to_datetime(df.SCH_DEP_DTZ) 
    #df.SCH_ARR_TMZ = df.SCH_ARR_TMZ.str[0:5]
    #df.SCH_ARR_DTZ = pd.to_datetime(df.SCH_ARR_DTZ)


    # Remove extraneous blank space
    #try: df['FOD'] = df['FOD'].str.strip()
    #except: pass
    #try: df['FOD_REAL'] = df['FOD_REAL'].str.strip()
    #except: pass
    #df['OD'] = df['OD'].str.strip()
    #try: df['ORIGIN_PLANNED'] = df['ORIGIN_PLANNED'].str.strip()
    #except: pass
    #df['ORIG_CD'] = df['ORIG_CD'].str.strip()
    #df['DEST_CD'] = df['DEST_CD'].str.strip()
    
    try: df['COUNT_ATB'] = df['COUNT_ATB'].astype('int')
    except: pass
    try: df['COUNT_GTB'] = df['COUNT_GTB'].astype('int')
    except: pass
    
    try: df['COUNT_DIVERSION'] = df['COUNT_DIVERSION'].astype('int')
    except: pass
    
    df['FLT_NUM'] = df['FLT_NUM'].astype('int')

    df['CAPACITY_CNT'] = df['CAPACITY_CNT'].astype('int')
    df['CREW_CNT'] = df['CREW_CNT'].fillna(0).astype('float').astype('int')
    df['CAPACITY_C_CNT'] = df['CAPACITY_C_CNT'].astype('int')
    df['CAPACITY_Y_CNT'] = df['CAPACITY_Y_CNT'].astype('int') # Drop ETD_DTMZ: 93% of rows are '?' df = df.drop(labels='ETD_DTMZ', axis=1)
    df['CANCEL_CD'] = df['CANCEL_CD'].fillna(np.nan).astype(float)

    # CM, KL, UA, etc
    # No longer exists in the table
    #df.AC_OWNER_CARRIER_CD = df.AC_OWNER_CARRIER_CD.str.strip()

    """
    df.IN_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.IN_DTZ = pd.to_datetime(df.OUT_DTMZ)
    df.OUT_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.OUT_DTZ = pd.to_datetime(df.OUT_DTMZ)
    print("df.OUT_DTZ")  # not in return statement. BUG !!!
    print(df.OUT_DTZ)
    df.ON_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.ON_DTZ = pd.to_datetime(df.ON_DTMZ)
    df.OFF_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.OFF_DTZ = pd.to_datetime(df.OFF_DTMZ)
    """

    # Drop all labels ending in DTL, DTML, TML
    # Convert df.columns to dataframe to allow string manipulation
    # to_frame: convert to dataframe
    # reset index: transfer index to column
    ix = df.columns.to_frame('index').reset_index()['index']
    ixd1 = ix[ix.str.contains('DTML')].index.values.tolist()
    ixd2 = ix[ix.str.contains('_DTL')].index.values.tolist()
    ixd3 = ix[ix.str.contains('_TML')].index.values.tolist()
    idx = list(set(ixd1+ixd2+ixd3)) #+ixd4+ixd5+ixd6))
    print("drop following columns: ", df.columns[idx])

    # UNCOMMENT ONCE PLOTTING in timezones IS DEBUGGED
    # df = df.drop(labels=df.columns[idx], axis=1)

    # Identify columns with DTMZ dates, and store them in integer format. 
    # This integer measures nanoseconds since a fixed starting date
    # The column can efficiently convert back to datetime using the pandas 
    # function pd.to_datetime()

    ix = df.columns.to_frame('index').reset_index()['index']
    ixd1 = ix[ix.str.contains('DTMZ')].index.values.tolist()
    idx = list(set(ixd1)) 
    cols = list(df.columns[idx])
    cols.append('SCH_DEP_DTML_PTY')
    cols.append('SCH_ARR_DTML_PTY')

    #print(getSortedHourCount(df['SCH_ARR_DTML_PTY'], msg="cleanup: SCH_ARR_DTML_PTY"))
    #return 

    ## Somehow, there was a screwup in this loop (col index)
    #---------------
    tmz = series_to_time_components(df['SCH_ARR_DTML_PTY'])
    #print(">> bef tmz: ",  tmz['h'].value_counts().sort_index().head(50) )
    #---------------

    # Convert datetime formats (DTMZ) to int formation (nanoseconds since a fixed time)
    # This format converts to datetime superfast  
    for col in cols:
        print("col: ", col)
        try:   # NaN cannot be converted to int
            df[col] = pd.to_datetime(df[col]).astype(int) # NaN -> NaT
        except:
            df[col] = df[col].replace(np.nan, '1960-01-01 00:01:00')
            df[col] = pd.to_datetime(df[col]).astype(int) # NaN -> NaT
        #print("col converted to int")

    df['SCH_DEP_DTZ'] = pd.to_datetime(df.SCH_DEP_DTMZ).dt.strftime("%Y/%m/%d")
    df['SCH_ARR_DTZ'] = pd.to_datetime(df.SCH_ARR_DTMZ).dt.strftime("%Y/%m/%d")
    df['SCH_DEP_TMZ'] = pd.to_datetime(df.SCH_DEP_DTMZ).dt.strftime("%H:%m")
    df['SCH_ARR_TMZ'] = pd.to_datetime(df.SCH_ARR_DTMZ).dt.strftime("%H:%m")

    #---------------
    #tmz = series_to_time_components(df['SCH_ARR_DTML_PTY'])
    #print(">> aft tmz: ",  tmz['h'].value_counts().sort_index().head(50) )
    #---------------

    #for col in ['OFF','ON','IN','OUT']:
        #df[col] = df[col].str[0:5]   # Limit time to hours/min

    # Based on the output of pandas_profiling, remove columns where all variables are identical 
    # Remove non-useful columns. Only keep times in DTMZ format. 
    try:
        df = df.drop(labels=['DEP_DELAY_INTERVAL','ARR_DELAY_INTERVAL','CANCEL_REASON_DESCRIPTION', 'FOD','FOD_REAL','OPERATED_ALL','OPERATED_CNT_ALL','ROTATION_@STATION', 'WEEK','YEAR','MONTH','Q','FUENTE'], axis=1)
        # Best to do my own operations if needed on DTMZ fields
        df = df.drop(labels=['OFF','ON','IN','OUT'], axis=1)
    except:
        pass
    
    return [df, non_converted]

#------------------------------------------------------
def cleanupColumns(df1):
    """
    Cleanup the columns of the FSU tables provided by Copa in Jan. 2021.
    Only applies to new FSU table until code is debugged

    Parameters: 
        df1: DataFrame to clean

    Return: a tuple with two objects: 
       df: the cleaned dataframe
       non_convertad: a list of column names that are not strings and could not be converted to 'float' or 'int'
       

    Parameters: DataFrame to clean
    No columns are deleted. Character columns are stripped (blank spaces removed in front and back). 
    Some columns are converted to int. 
    Time columns (DTMZ) only keep hours and minutes. 
    Date columns are changed so that the dates are sortable (yyyy/mm/dd) as strings. 

    Return: modified dataframe.
    """
    df = df1.copy()
    non_converted = []

    # Columns with special treatment
    df['ARR_DELAY_MINUTES'] = df['ARR_DELAY_MINUTES'].replace('?', 999999).astype(float).astype(int)
    df['DEP_DELAY_MINUTES'] = df['DEP_DELAY_MINUTES'].replace('?', 999999).astype(float).astype(int)

    # Replace all other '?' by NaN
    df.replace('?', np.nan, inplace=True)
        
    # Strip 'object' or 'str' types
    # Convert other columns to integers after converting NaN
    for col in df.columns:
        if df[col].dtype == 'O' or df[col].dtype == 'str':
            df[col] = df[col].str.strip()
        else:
            try:
                pass
                df[col] = df[col].astype('int')
            except:
                try:
                    df[col] = df[col].fillna(-1).astype('int')
                    df[col] = df[col].astype('int')
                except:
                    non_converted.append(col)
    # string format
    df.SCH_DEP_TMZ = df.SCH_DEP_TMZ.str[0:5]
    # datetime format
    df.SCH_DEP_DTZ = pd.to_datetime(df.SCH_DEP_DTZ) 
    df.SCH_ARR_TMZ = df.SCH_ARR_TMZ.str[0:5]
    df.SCH_ARR_DTZ = pd.to_datetime(df.SCH_ARR_DTZ)


    # Remove extraneous blank space
    try: df['FOD'] = df['FOD'].str.strip()
    except: pass
    try: df['FOD_REAL'] = df['FOD_REAL'].str.strip()
    except: pass
    df['OD'] = df['OD'].str.strip()
    try: df['ORIGIN_PLANNED'] = df['ORIGIN_PLANNED'].str.strip()
    except: pass
    df['ORIG_CD'] = df['ORIG_CD'].str.strip()
    df['DEST_CD'] = df['DEST_CD'].str.strip()
    
    try: df['COUNT_ATB'] = df['COUNT_ATB'].astype('int')
    except: pass
    try: df['COUNT_GTB'] = df['COUNT_GTB'].astype('int')
    except: pass
    
    try: df['COUNT_DIVERSION'] = df['COUNT_DIVERSION'].astype('int')
    except: pass
    
    df['FLT_NUM'] = df['FLT_NUM'].astype('int')

    df['CAPACITY_CNT'] = df['CAPACITY_CNT'].astype('int')
    df['CREW_CNT'] = df['CREW_CNT'].fillna(0).astype('float').astype('int')
    df['CAPACITY_C_CNT'] = df['CAPACITY_C_CNT'].astype('int')
    df['CAPACITY_Y_CNT'] = df['CAPACITY_Y_CNT'].astype('int') # Drop ETD_DTMZ: 93% of rows are '?' df = df.drop(labels='ETD_DTMZ', axis=1)
    df['CANCEL_CD'] = df['CANCEL_CD'].fillna(np.nan).astype(float)

    # CM, KL, UA, etc
    df.AC_OWNER_CARRIER_CD = df.AC_OWNER_CARRIER_CD.str.strip()

    """
    df.IN_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.IN_DTZ = pd.to_datetime(df.OUT_DTMZ)
    df.OUT_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.OUT_DTZ = pd.to_datetime(df.OUT_DTMZ)
    print("df.OUT_DTZ")  # not in return statement. BUG !!!
    print(df.OUT_DTZ)
    df.ON_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.ON_DTZ = pd.to_datetime(df.ON_DTMZ)
    df.OFF_TMZ = df.SCH_DEP_DTMZ.str[-12:-7]
    df.OFF_DTZ = pd.to_datetime(df.OFF_DTMZ)
    """

    # Drop all labels ending in DTL, DTML, TML
    # Convert df.columns to dataframe to allow string manipulation
    # to_frame: convert to dataframe
    # reset index: transfer index to column
    ix = df.columns.to_frame('index').reset_index()['index']
    ixd1 = ix[ix.str.contains('DTML')].index.values.tolist()
    ixd2 = ix[ix.str.contains('_DTL')].index.values.tolist()
    ixd3 = ix[ix.str.contains('_TML')].index.values.tolist()
    idx = list(set(ixd1+ixd2+ixd3)) #+ixd4+ixd5+ixd6))
    print("drop following columns: ", df.columns[idx])

    # UNCOMMENT ONCE PLOTTING in timezones IS DEBUGGED
    # df = df.drop(labels=df.columns[idx], axis=1)

    # Identify columns with DTMZ dates, and store them in integer format. 
    # This integer measures nanoseconds since a fixed starting date
    # The column can efficiently convert back to datetime using the pandas 
    # function pd.to_datetime()

    ix = df.columns.to_frame('index').reset_index()['index']
    ixd1 = ix[ix.str.contains('DTMZ')].index.values.tolist()
    idx = list(set(ixd1)) 
    cols = list(df.columns[idx])
    cols.append('SCH_DEP_DTML_PTY')
    cols.append('SCH_ARR_DTML_PTY')

    #print(getSortedHourCount(df['SCH_ARR_DTML_PTY'], msg="cleanup: SCH_ARR_DTML_PTY"))
    #return 

    ## Somehow, there was a screwup in this loop (col index)
    #---------------
    tmz = series_to_time_components(df['SCH_ARR_DTML_PTY'])
    #print(">> bef tmz: ",  tmz['h'].value_counts().sort_index().head(50) )
    #---------------

    # Convert datetime formats (DTMZ) to int formation (nanoseconds since a fixed time)
    # This format converts to datetime superfast  
    for col in cols:
        print("col: ", col)
        try:   # NaN cannot be converted to int
            df[col] = pd.to_datetime(df[col]).astype(int) # NaN -> NaT
        except:
            df[col] = df[col].replace(np.nan, '1960-01-01 00:01:00')
            df[col] = pd.to_datetime(df[col]).astype(int) # NaN -> NaT
        #print("col converted to int")

    #---------------
    tmz = series_to_time_components(df['SCH_ARR_DTML_PTY'])
    #print(">> aft tmz: ",  tmz['h'].value_counts().sort_index().head(50) )
    #---------------

    for col in ['OFF','ON','IN','OUT']:
        df[col] = df[col].str[0:5]   # Limit time to hours/min

    # Based on the output of pandas_profiling, remove columns where all variables are identical 
    # Remove non-useful columns. Only keep times in DTMZ format. 
    try:
        df = df.drop(labels=['DEP_DELAY_INTERVAL','ARR_DELAY_INTERVAL','CANCEL_REASON_DESCRIPTION', 'FOD','FOD_REAL','OPERATED_ALL','OPERATED_CNT_ALL','ROTATION_@STATION', 'WEEK','YEAR','MONTH','Q','FUENTE'], axis=1)
    except:
        pass
    #print(df.columns)
    
    return [df, non_converted]

#------------------------------------------------------
def to_datetime(series):
    series_tmz = series.str[-12:-7]
    series_dtz = pd.to_datetime(series)
    return pd.concat([series_dtz, series_tmz], axis=1)
#------------------------------------------------------
def additionalCleanup(df):
    """
    Additional cleanup beyond cleanupColumns

    Parameters: 
        df (DataFrame)

    return: 
        Modified dataframe

    Execute the following operations: remove cancellations, turnbacks, diversions, and flights not in [1,999]
    """

    df = removeCancellations(df)
    df = removeTurnbacks(df)
    df = removeDiversions(df)
    df = filterFlights(df)
    return df

#----------------------------------------------------------
def saveFullCleanup(df, out_filenm):
    df, _ = cleanupColumns(df)
    df = additionalCleanup(df)
    df.to_csv(out_filenm, index=0)
    print("Clean data saved to file: ", out_filenm)

#----------------------------------------------------------
def displayCancellations(df):
    cols4 = ['SCH_DEP_DTZ','SCH_DEP_TMZ','FLT_NUM','CLAVLOCOD','FLT_TYPE_CD','FLT_TYPE_NAME','COUNT_DIVERSION','HUB_STN','OD','ORIG_CD','DEST_CD','CANCEL_CD','COUNT_ATB','COUNT_GTB']
    display(df['CANCEL_CD'].unique())
    display(df[df['CANCEL_CD'] == 1][cols4].head(20))  


def listDiversions(df):
    div = df[df['FLT_TYPE_NAME'].str.contains('Diversion')].sort_values(['SCH_DEP_DTZ','SCH_DEP_TMZ'])
    cols1 = ['OD','FOD_REAL','FOD','ORIGIN_PLANNED','COUNT_DIVERSION','COUNT_GTB','COUNT_ATB','CANCEL_CD','CLAVLOCOD']
    cols2 = ['SCH_DEP_DTMZ','FLT_TYPE_CD','CLAVLOCOD','DEP_DELAY_MINUTES','COUNT_DIVERSION']
    cols3 = ['SCH_DEP_DTZ','SCH_DEP_TMZ','FLT_NUM']
    # FOD is always FLT_NUM + OD
    # FOD_REAL is always FLT_NUM + (ORIG_CD+DEST_CD)
    cols4 = ['SCH_DEP_DTZ','SCH_DEP_TMZ','FLT_NUM','CLAVLOCOD','FLT_TYPE_CD','FLT_TYPE_NAME','COUNT_DIVERSION','HUB_STN','OD','ORIG_CD','DEST_CD','COUNT_ATB','COUNT_GTB']
    #print("diversions: "); display(div[cols4].head(300))
    # Print FSU on 10/6
    d = fsu[fsu['COUNT_DIVERSION']==1].sort_values(['SCH_DEP_DTZ','SCH_DEP_TMZ'])
    f = fsu[fsu['SCH_DEP_DTZ']=='2019-10-01'].sort_values(['SCH_DEP_DTZ','SCH_DEP_TMZ'])
    i = fsu[fsu['HUB_STN'].str.strip()=='INTRACAM'].sort_values(['SCH_DEP_DTZ','SCH_DEP_TMZ'])
    print(fsu.groupby('HUB_STN').size()) #.unique())
    nb_listed = 5
    print("INTRACAM: "); display(i[cols4].head(nb_listed))
    print("count_diversions: "); display(d[cols4].head(nb_listed))
    print("diversions: "); display(f[cols4].head(nb_listed))
    print("CLAVLOCOD: "); display(df['CLAVLOCOD'].unique())
    print("FLT_TYPE_CD: "); display(df['FLT_TYPE_CD'].unique())
    #print("diversions: ", div[cols4])




def listSpecialFlights(df):
    fod_real = df['FOD_REAL'].str[-6:]
    fod = df['FOD'].str[-6:]
        
    od = df['OD']
    dest = df['DEST_CD']
    orig = df['ORIG_CD']
    dtz = df['SCH_DEP_DTZ']
    tmz = df['SCH_DEP_TMZ']
    origdest = orig+dest
    atb = (df['COUNT_ATB'] > 0)
    gtb = (df['COUNT_GTB'] > 0)

    flt_type = (df['FLT_TYPE_CD'])
    flt = df['FLT_NUM']
    dep_delay = df['DEP_DELAY_MINUTES']
    
    print("flt= ", df['FLT_NUM'])
 
    count_diversion = (df['COUNT_DIVERSION'])
    print("count_diversion_sum= ", count_diversion.sum())

    s = df.groupby('COUNT_DIVERSION').size()
    print("\nsum(COUNT_DIVERSION)= \n", s, "\n")
    
    diversions = (df['CLAVLOCOD'] != 'T')
    print("nb diversions: ", diversions.sum())
    
    s = df.groupby('CLAVLOCOD').size()
    print("\nCLAVLOCOD: \n", s, "\n")

    s = df.groupby('FLT_TYPE_CD').size()
    print("\nFLIGHT_TYPE_CD: \n", s, "\n")
    
    s = df.groupby('FLT_TYPE_NAME').size()
    print("\nFLIGHT_TYPE_NAME: \n", s, "\n")
    
    s = df.groupby('HUB_STN').size()
    print("\nHUB_STN: \n", s, "\n")
    
    print("nb atb: ", atb.sum())
    print("nb gtb: ", gtb.sum())
    
    origdest_od = (origdest != od)
    print('origdest != od: ', origdest_od.sum())
    
    fod_fodreal = (fod != fod_real)
    print('fod != fod_real: ', fod_fodreal.sum())
    
    fod_od = (fod != od)
    print("fod != od: ", fod_od.sum())
    
    fod_real_origdest = (fod_real != origdest)
    print("fod_real != origdest: ", fod_real_origdest.sum())

    cols1 = ['FOD_REAL','FOD','OD','ORIGIN_PLANNED','COUNT_DIVERSION','COUNT_GTB','COUNT_ATB','CANCEL_CD','CLAVLOCOD','HUB_STN']
    cols2 = ['SCH_DEP_DTMZ','FLT_TYPE_CD','CLAVLOCOD','DEP_DELAY_MINUTES','COUNT_DIVERSION']
    cols3 = ['SCH_DEP_DTZ','SCH_DEP_TMZ','FLT_NUM']
    
    flts = df.groupby(cols3).size().to_frame('size')
    print("max nb flts: ", flts.max())
    # which flights are these? 
    print("fltsz3: "); display(flts[flts['size'] == 3])
    print("fltsz2: "); display(flts[flts['size'] == 2])
    #1/11/2020   23:43:00.000000 679.0       3

    sz3 = df[(dtz=='1/11/2020') & (tmz=='23:43:00.000000') & (flt == 679)]
    print("\n>> size 3 flights: \n"); display(sz3[cols3])
    print("\n>> size 3 flights: \n"); display(sz3[cols2])
    print("\n>> size 3 flights: \n"); display(sz3[cols1])
    
    sz3 = df[(dtz=='10/24/2019') & (tmz=='05:36:00.000000') & (flt == 123)]
    print("\n>> size 3 flights: \n"); display(sz3[cols2])
    print("\n>> size 3 flights: \n"); display(sz3[cols1])
    
    sz3 = df[(dtz=='12/21/2019') & (tmz=='11:10:00.000000') & (flt == 658)]
    print("\n>> size 3 flights: \n"); display(sz3[cols2])
    print("\n>> size 3 flights: \n"); display(sz3[cols1])
    
    #2/4/2020    05:14:00.000000 477  
    sz2 = df[(dtz=='2/4/2020') & (tmz=='05:14:00.000000') & (flt == 477)]
    print("\n>> size 2 flights: \n"); display(sz2[cols2])
    print("\n>> size 2 flights: \n"); display(sz2[cols1])
    
    # 10/15/2019  23:36:00.000000 192         2
            
    sz2 = df[(dtz=='10/15/2019') & (tmz=='23:36:00.000000') & (flt == 192)]
    print("\n>> size 2 flights: \n"); display(sz2[cols2])
    print("\n>> size 2 flights: \n"); display(sz2[cols1])

    print("\n>> Diversions (CLAVLOCOD != T)"); display(fsu.loc[diversions,cols2].head(10))
        
    # When FOD != FOD_REAL, the flight is a diversion
    print("\n>> fod_real != fod"); display(fsu[fod_real != fod][cols1].head(10))
    
    # THERE ARE TWO 'CLAVLOCOD' left that are 'T' (regular flight). WHY???
    f = fsu[fod_real != fod][cols1]
    display(f[(f['CLAVLOCOD'] != 'C') & (f['CLAVLOCOD'] != 'D')])
    #display(fsu)




def filterFlights(df):
    df['FLT_NUM'] = df['FLT_NUM'].astype(int)
    df = df[(df['FLT_NUM'] > 0) & (df['FLT_NUM'] < 1000)]
    #df['FLT_NUM'] = df['FLT_NUM'].copy().astype(str)
    return df




def filterCarriers(df):
    # AC_OWNER_CARRIER_CD no longer in the FSU table
    # return df
    print("keepCM")
    print(df.shape)
    try:
        df.AC_OWNER_CARRIER_CD = df.AC_OWNER_CARRIER_CD.str.strip()
        # (this keeps CM* as well)
        df = df[(df['AC_OWNER_CARRIER_CD'] == 'CM')]
        print(df.shape)
    except:
        pass
    return df




def removeCancellations(df):
    print("removeCancellations")
    #df = pd.read_csv("df_fsu_clean.csv")
    df['CANCEL_CD'] = df['CANCEL_CD'].replace('?',-1)
    df['CANCEL_CD'] = df['CANCEL_CD'].fillna(-1)
    df['CANCEL_CD'] = df['CANCEL_CD'].astype('int') 
    df = df[df['CANCEL_CD'] == 0]   # Remove cancellations
    print("df.shape: ", df.shape)
    return df




def removeDiversions(df):
    print("removeDiversions", df.shape)

    # Count Diversion counts the number of diversions, but is not set to 
    # one on each leg of a diversion
    if 'COUNT_DIVERSION' in df.columns:
        nodiversion = (df['COUNT_DIVERSION'] == 0)
        df = df.loc[nodiversion,:]
        df = df.drop(labels='COUNT_DIVERSION', axis=1)

    # Strangely, COUNT_DIVERSION does not capture all diversions
    # The next line captures the remaining diversions
    #print((df['OD'] == '?').sum())
    #print((df['ORIG_CD'] == '?').sum())
    #print((df['DEST_CD'] == '?').sum())
    nodiversion = (df['OD'] == (df['ORIG_CD']+df['DEST_CD']))
    df = df.loc[nodiversion,:]

    ODX = df['ORIG_CD'] + df['DEST_CD']
    nodiversion = (df['OD'] != ODX)
    print("no diversions: ")
    print(df.loc[nodiversion,:].shape)


    print(df.shape)
    return df




def removeTurnbacks(df):
    print("removeTurnbacks")
    if 'COUNT_ATB' in df.columns:
        df = df[df['COUNT_ATB'] == 0].copy()
    if 'COUNT_GTB' in df.columns:
        df = df[df['COUNT_GTB'] == 0].copy()
    # remove columns since the column has constant values
    dfmex = df[df['OD'] == 'MEXMEX'] # <<<<< WILL THIS PRINT? 
    print(dfmex)
    try:  # in case the column does not exist
        df = df.drop(labels=['COUNT_ATB','COUNT_GTB'], axis=1)
    except:
        pass
    print(df.shape)
    return df




def removeOeqD(df):
    print("removeOeqD:", df.shape)
    try:
        df = df[(df['ORIG_CD'] != df['DEST_CD'])]
    except:
        pass
    print(df.shape)
    return df




def relabelColumnsFSU(fsu):
    print("relabelColumns, ", fsu.shape)
    #fsu = pd.read_csv("df_fsu_clean_flt1-999.csv")
    #print(fsu.columns)
    
    # Remove columns AC_OWNER_CARRIER_CD and AIRCRAFT_TYPE_ICAO since 
    # they are not in the FSU database: 10/1/2019 - 6/1/2021 (1.5 years)
    cols_upper = ['OFF_DTMZ','ON_DTMZ','CAPACITY_CNT','OD','CAPACITY_C_CNT','CAPACITY_Y_CNT','SCH_DEP_DTZ','SCH_DEP_TMZ','FLT_NUM','ORIG_CD','DEST_CD','CREW_CNT','AIRCRAFT_TYPE','ROTATION_AVAILABLE_TM','ROTATION_PLANNED_TM','ROTATION_REAL_TM','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES','TAXI_IN_MINUTES','TAXI_OUT_MINUTES','IN_DTMZ','OUT_DTMZ']

    cols_lower = ['offz','onz','capacity','od','c_cnt','y_cnt','dep_dtz','dep_tmz','flt_num','orig','dest','crew','air_type','rot_avail','rot_plan','rot_real','arr_delay','dep_delay','taxi_in','taxi_out','inz','outz']

    fsu_upper = fsu[cols_upper]
    fsu_lower = fsu_upper.copy()
    #fsu_lower.columns = cols_lower
    return fsu_upper, fsu_lower;

# -----------------------------------


def checkUniques(df, max_unique=200, print_unique=30):
    cols = df.columns
    for i,col in enumerate(cols):
        n = df[col].nunique()
        typ = df[col].dtype
        if typ != 'float64':
            print(f"({i},{typ}) ", col, f"({n})", df[col].unique()[:print_unique])




def checkColFloat(df, col_nm):
    v = df[col_nm].values
    failed = []
    for i,el in enumerate(v):
        try:
            a = float(el)
        except:
            failed.append((i,el))
            continue
    return failed




def nbNonFloats(df):
    cols = df.columns
    for col in cols:
        nb_non_floats = 0
        non_floats = []
        for el in df[col]:
            try:
                float(el)
            except:
                nb_non_floats += 1
                non_floats.append(el)
        print(col, nb_non_floats)



def makeFloatDf(df):
    dff = df.copy()

    for col in dfx.columns:
        dff[col] = dfxx[col]
    return dff

# ============================================
# Feeder Functions

def connectingFromPaxStatistics(df):
    # Given a collection of records with _f and _nf column headers, for each non-feeder flight (_nf),
    # compute the PAX embarking to each _nf flight leaving PTY.
    # Also compute the number in C and Y classes
    # By grouping over many variables, it is easy to aggregate with respect to any subset.
    dfg = df.groupby(['dep_dtz_nf','dep_tmz_nf','flt_num_nf','cabin_nf']).sum()
    return dfg

def connectingToPaxStatistics(df):
    # Given a collection of records with _f and _nf column headers, for each feeder flight (_f) arriving
    # at PTY, compute the PAX embarking to all non-feeder flights (_nf) departing from PTY.
    # Also compute the number in C and Y classes
    # By grouping over many variables, it is easy to aggregate with respect to any subset.
    dfg = df.groupby(['arr_dtz_f','arr_tmz_f','flt_num_f','cabin_f']).sum()
    return dfg

#------------------------------------------------
def plotFeederPaxByDates(df_list, ylim=None):
    """
    Plot a list of dataframes aa a function of date

    df_list: list of of two dataframes to plot as a function of the day
       Group by 'dep_dtz_nf', sum the pax
       Group by 'arr_dtz_f',  sum the pax from feeders
    The plot has vertical bars to separate the months 
    the month labels are not visible on the plot
    Hardcoded full range: Sept 2019 - Feb 2020
    The following fields should be available: 
    'dep_dtz_nf, 'pax_nf', 'arr_dtz_f'
    'labels are also hardcoded
    """

    fig, ax = plt.subplots(1, 1)
    dfgfrom, dfgto = df_list

    pax_from = dfgfrom.groupby(level='dep_dtz_nf')['pax_f'].sum()
    pax_to = dfgto.groupby(level='arr_dtz_f')['pax_nf'].sum()

    sns.set(rc={'figure.figsize':(15,6.)})
    nb_days = -1
    dates_ = pax_from.index[0:nb_days].str.replace('/','-')

    #dates = pd.to_datetime(pax_from.index[0:nb_days], unit='D', origin=pd.Timestamp('2019-01-01'), format = '%Y/%m/%d')
    # ORIGIN IS SCREWED UP. I WANT TO SET IT
    #d1 = pd.todatetime(  pd.Timestamp(date(2020,4,23))  )
    #print(d1)
    dates = pd.to_datetime(dates_, infer_datetime_format=True) #, format = '%Y/%m/%d')

    ax.plot_date(pax_from.index[0:nb_days], pax_from[0:nb_days], color="blue", ms=3, label="PAX_from", linestyle="-")
    ax.plot_date(pax_to.index[0:nb_days],   pax_to[0:nb_days],   color="red",    ms=3, label="PAX_to", linestyle="-")

    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])

    # Draw vertical lines at month junctures
    dates = pax_from.index[0:nb_days]

    # specify the position of the major ticks at the beginning of the week
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))     #### <<<< md doesnot exist !!!!! BUG BUG BUG
    # specify the format of the labels as 'year/month/day'
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y/%m/%d'))
    # specify the position of the minor ticks at each day
    ax.xaxis.set_minor_locator(md.DayLocator(interval = 1)) # every 7 days
    # (optional) rotate by 90° the labels in order to improve their spacing
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90.)   # WRONG ROTATION!!!

    # Format the x-axis for dates (label formatting, rotation)
    #fig.autofmt_xdate(rotation=90.)
    # Control tick lengths (does not work)
    ax.tick_params(axis = 'x', which = 'major', length = 5)
    ax.tick_params(axis = 'x', which = 'minor', length = 2)
    xlim = ax.get_xlim();
    lg = xlim[1] - xlim[0]
    # Figure out monthly boundaries
    monthly = [0., 30., 60., 90., 120., 150.,180.,210.] # 30 days per month
    xcoord = []
    midpoint = []
    nb_dates = len(dates)
    single_month_x = (lg / nb_dates) * 30.5 # 30.5 is average length of one month
    for i,month in enumerate(monthly):
        xm = xlim[0] + i * single_month_x
        xcoord.append(xm)
        ax.axvline(x=xm, ymax = 8400.)
    for i in range(0,len(monthly)-1):
        midpoint.append(0.5*(xcoord[i]+xcoord[i+1]))
    # Set xlim sligtly beyond in/max so that monthly boundary is visible
    ax.set_xlim(xlim[0]-1, xlim[1]+1)
    #ax.set_xticks(rotation=70)   # DOES NOT WORK
    ax.set_ylabel("Connecting PAX", fontsize=14)
    ax.set_xlabel("Departure day (9/1/2019 - 3/1/2020)", fontsize=14)
    labels = ['Sept. 2019', 'Oct. 2019', 'Nov. 2019', 'Dec. 2019', 'Jan. 2020', 'Feb. 2020']
    for i in range(0,len(monthly)-1):
        try:
            ax.text(midpoint[i]-5,6000,labels[i])
        except:
            pass
    plt.legend(fontsize=16)
    plt.gcf().autofmt_xdate()
#--------------------------------------------------
def plotPaxByDate(df_list, labels, title, ymonth=8000, ylim=None):
    """
    Plot a list of DataSeries aa a function of date

    df_list: list DataSeries to plot as a function of the day
    The plot has vertical bars to separate the months 
    the month labels are not visible on the plot
    Hardcoded full range: Sept 2019 - Feb 2020
    The following fields should be available: 
    'dep_dtz_nf, 'pax_nf', 'arr_dtz_f'
    'labels are also hardcoded
    """

    fig, ax = plt.subplots(1, 1)

    colors = ["red", "blue", "green", "cyan"]

    sns.set(rc={'figure.figsize':(15,6.)})
    nb_days = -1

    df = df_list[0]
    dates_ = df.index[0:nb_days].str.replace('/','-')

    #dates = pd.to_datetime(pax_from.index[0:nb_days], unit='D', origin=pd.Timestamp('2019-01-01'), format = '%Y/%m/%d')
    # ORIGIN IS SCREWED UP. I WANT TO SET IT
    #d1 = pd.todatetime(  pd.Timestamp(date(2020,4,23))  )
    #print(d1)
    dates = pd.to_datetime(dates_, infer_datetime_format=True) #, format = '%Y/%m/%d')

    for i, df in enumerate(df_list):
        ax.plot_date(df.index[0:nb_days], df[0:nb_days], color=colors[i], ms=3, label=labels[i], linestyle="-")
    
    if ylim != None:
        ax.set_ylim(ylim[0], ylim[1])

    # Draw vertical lines at month junctures
    dates = df_list[0].index[0:nb_days]

    # specify the position of the major ticks at the beginning of the week
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=1))
    # specify the format of the labels as 'year/month/day'
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y/%m/%d'))
    # specify the position of the minor ticks at each day
    ax.xaxis.set_minor_locator(md.DayLocator(interval = 1)) # every 7 days
    # (optional) rotate by 90° the labels in order to improve their spacing
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60.)   # WRONG ROTATION!!!

    # Format the x-axis for dates (label formatting, rotation)
    #fig.autofmt_xdate(rotation=90.)
    # Control tick lengths (does not work)
    ax.tick_params(axis = 'x', which = 'major', length = 5)
    ax.tick_params(axis = 'x', which = 'minor', length = 2)
    xlim = ax.get_xlim();
    lg = xlim[1] - xlim[0]
    # Figure out monthly boundaries
    monthly = [0., 30., 60., 90., 120., 150.,180.,210.] # 30 days per month
    xcoord = []
    midpoint = []
    nb_dates = len(dates)
    single_month_x = (lg / nb_dates) * 30.5 # 30.5 is average length of one month

    for i,month in enumerate(monthly):
        xm = xlim[0] + i * single_month_x
        xcoord.append(xm)
        ax.axvline(x=xm, ymax = 8400.)

    for i in range(0,len(monthly)-1):
        midpoint.append(0.5*(xcoord[i]+xcoord[i+1]))

    # Set xlim sligtly beyond in/max so that monthly boundary is visible
    ax.set_xlim(xlim[0]-1, xlim[1]+1)
    #ax.set_xticks(rotation=70)   # DOES NOT WORK
    ax.set_ylabel("Connecting PAX", fontsize=14)
    ax.set_xlabel("Departure day (9/1/2019 - 3/1/2020)", fontsize=14)
    labels = ['Sept. 2019', 'Oct. 2019', 'Nov. 2019', 'Dec. 2019', 'Jan. 2020', 'Feb. 2020']
    for i in range(0,len(monthly)-1):
        try:
            ax.text(midpoint[i]-5,ymonth,labels[i])
        except:
            pass
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    #plt.gcf().autofmt_xdate()
#----------------------------------------
def plotOD(ax, OD, df_od_group):
    #asu_pty = dffCMgr.get_group(('ASU','PTY'))
    # Remove NaN (expensive)
    df_od_group_clean = df_od_group.dropna(axis=0) # drop rows with NaNs
    #print(df_od_group_clean.columns)
    arr = df_od_group_clean.ARR_DELAY_MINUTES.values
    dep = df_od_group_clean.DEP_DELAY_MINUTES.values
    ax.hist(arr, bins=50, alpha=.5, color='r', range=[-50, 50], label='Arrival')
    ax.hist(dep, bins=50, alpha=.5, color='b', range=[-50, 50], label='Departure')
    # plot mean of each on the same plot
    mean_arr = arr.mean()
    mean_dep = dep.mean()
    mx_arr = np.max(arr)
    mn_arr = np.min(arr)
    mx_dep = np.max(dep)
    mn_dep = np.min(dep)

    count = arr.shape[0]
    try:
      ax.scatter([mean_arr], [0.], color='r', marker='o', s=20, lw=2,
        label=f"{int(mean_arr)} min")
      ax.scatter([mean_dep], [0.], color='b', marker='o', s=20, lw=2,
        label=f"{int(mean_dep)} min")
      ax.set_xlim(-50, 50)
    except:
        print("*****")
        print(df_od_group)

    ax.legend(fontsize=6)
    subtitle = f"dep: %.0f/%.0f, arr: %.0f/%.0f" % (mn_dep,mx_dep,mn_arr,mx_arr)
    ax.set_title(f"{OD[0]}-{OD[1]}, {count} flts\n%s" % subtitle, fontsize=8)
#------------------------------------------
def plotDelaysAllFlights():
    nr, nc = 21,5
    fig, axes = plt.subplots(nr, nc, figsize=(12,40))
    axf = np.asarray(axes).reshape(-1)
    keys = list(dffCMgr.groups.keys())

    i = 0
    for key in keys:
        #for i, grp in enumerate(dffCMgr.groups.keys()):
        #print(f"{i}, {key}")
        grp = dffCMgr.get_group(key)
        count = grp.shape[0]
        #print("count= ", count, key)
        if count < 125: continue
        #if i >= 10: break
        #if i >= nr*nc: break
        ax = axf[i]
        i += 1
        # print(grp.ORIG_CD.iloc[0])
        #print(dffCMgr.get_group(grp))
        plotOD(ax, key, grp)
    plt.tight_layout()
    plt.savefig("OD.pdf");
    plt.close()
    print("plot saved and closed")
#----------------------------------------
def nonFeederDepartures(df, feeders=False):
    """
    Compute number of flights and number of days in the data frame.

    nonFeederDepartures(df, feeders=False)
    df : data frame with columns: :dep_dtz_nf, :dep_tmz_nf, :flt_num_nf,
        :dep_dtz_f, :dep_tmz_f, :flt_num_f
    feeders : whether this is computed for feeders or non-feeders
    returns nb_days, nb_flt_num (number of days for which there are flights, and 
       the number of different flight numbers)
    """
    if not feeders:
        dfg = df.groupby(['dep_dtz_nf','dep_tmz_nf','flt_num_nf'])
        sz = dfg.size()
        nb_days_nf = sz.to_frame('size').groupby('dep_dtz_nf').size().shape
        nb_flt_num_nf = sz.to_frame('size').groupby('flt_num_nf').size().shape
        print("nb_days_nf: " ,nb_days_nf[0])
        print("nb_flt_num_nf: ", nb_flt_num_nf[0])
        return nb_days_nf, nb_flt_num_nf
    else:
        dfg = df.groupby(['dep_dtz_f','dep_tmz_f','flt_num_f'])
        sz = dfg.size()
        nb_days_f = sz.to_frame('size').groupby('dep_dtz_f').size().shape
        nb_flt_num_f = sz.to_frame('size').groupby('flt_num_f').size().shape
        print("nb_days_f: " ,nb_days_f[0])
        print("nb_flt_num_f: ", nb_flt_num_f[0])
        return nb_days_f, nb_flt_num_f
#----------------------------------------
def scatter_DepArrDelays(df):
    fsg = df.groupby('OD')[['ACTUAL_BLOCK_HR','FLT_ACTUAL_HR','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES']].mean()
    plt.subplots(2,2)
    plt.subplot(2,2,1)
    sns.scatterplot(data=fsg,x='DEP_DELAY_MINUTES',y='ACTUAL_BLOCK_HR', label='DEP/Block')
    plt.subplot(2,2,2)
    sns.scatterplot(data=fsg,x='DEP_DELAY_MINUTES',y='FLT_ACTUAL_HR', label='DEP/FLT')
    plt.subplot(2,2,3)
    sns.scatterplot(data=fsg,x='ARR_DELAY_MINUTES',y='ACTUAL_BLOCK_HR', label='ARR/Block')
    plt.subplot(2,2,4)
    sns.scatterplot(data=fsg,x='ARR_DELAY_MINUTES',y='FLT_ACTUAL_HR', label='ARR/FLT')
    plt.show()
#-----------------------------------------
def scatter_ArrMinusDepDelay(df):
    fsg = df.groupby('OD')[['ACTUAL_BLOCK_HR','FLT_ACTUAL_HR','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES']].mean()
    fsg['arr_dep_delay'] = fsg['ARR_DELAY_MINUTES'] - fsg['DEP_DELAY_MINUTES']
    plt.subplots(2,1)
    plt.subplot(2,1,1)
    sns.scatterplot(data=fsg,x='arr_dep_delay',y='FLT_ACTUAL_HR', label='FLT_HR')
    plt.subplot(2,1,2)
    sns.scatterplot(data=fsg,x='arr_dep_delay',y='ACTUAL_BLOCK_HR', label='BLOCK_HR')
    #sns.legend()
    plt.show()
#----------------------------------------
def ArrDepFltTimeCorr(df):
    fsg = df.groupby('OD')[['ACTUAL_BLOCK_HR','FLT_ACTUAL_HR','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES']].mean()
    fsg['arr_dep_delay'] = fsg['ARR_DELAY_MINUTES'] - fsg['DEP_DELAY_MINUTES']

    #dfcorr = df[['ARR_DELAY_MINUTES','DEP_DELAY_MINUTES','arr_dep_delay','FLT_ACTUAL_HR','ACTUAL_BLOCK_HR']]
    dfcorr = fsg

    corr = dfcorr.corr(method='pearson')
    print("\nPearson (How linear is the relationship): \n")
    display(corr)

    corr = dfcorr.corr(method='kendall')
    print("Kendall: \n")
    display(corr)

    corr = dfcorr.corr(method='spearman')
    print("\nSpearman=1 means that one variable is a monotonic function of the other.")
    display("Spearman: \n")
    display(corr)
    print("\nIf Spearman is high and Pearson lower, that means there is a possibly \n\
    nonlinear monotonic relationship between the variables")

#------------------------------------------
def flights_nonfeeders(df):
    """
    Function flights_nonfeeders(df)

    Compute some non-feeder pax statistics based on TRUE_OD and OD comparisons.
    Calculate some 

    Parameters
    df (DataFrame) :  the dataframe must include columns 'od', 't_od' (TRUE_OD), 'pax'.
    Return a tuple: (non-feeders, feeders).
    """

    # Flights to exclude from consideration
    df1 = df.loc[(df['od'] == df['t_od']),:].copy()
    df2 = df.loc[(df['t_od'].str[0:3] == 'PTY'),:].copy()
    df3 = df.loc[(df['t_od'].str[3:6] == 'PTY'),:].copy()

    #print("df1,df2,df3 shape: ", df1.shape, df2.shape, df3.shape)
    #print("enter: df1.index.max= ", df1.index.max())
    #print("enter: df2.index.max= ", df2.index.max())
    #print("enter: df3.index.max= ", df3.index.max())

    # Merge via the index, keep all indices from both dataframes
    dfm = df1.merge(df2, how='outer', on='index')
    dfm = dfm.merge(df3, how='outer', on='index')
    
    #print("dfm: ", dfm.columns)
    #print("index: ", dfm['index'])
    #print("dfm.index.max= ", df.index.max())

    potential_feeders = df.set_index('index').drop(index=dfm['index'].values, axis=0)

    potential_feeders['ptyorig'] = 1
    potential_feeders.loc[potential_feeders['od'].str[3:6] == 'PTY', 'ptyorig'] = 0

    return dfm, potential_feeders
#----------------------------------------------------
def save_all_feeders(potential_feeders, output_fn="feeders_to_flights.csv"):
    # Count the number of rows for each route ('recloc','t_od')
    # Each recloc has 2 or 3 t_od (true_od): two-ways trip with perhaps one or more intermediate flights
    #   during the trip. One way trips have a single true_od
    p1 = potential_feeders.copy()

    # Remove all flight numbers == 0 and flight_numbers > 999
    p1 = p1[(p1['flt_num'] > 0) & (p1['flt_num'] < 1000)]
 
    # Remove all flight whose carriers are not CM
    # There is no delay information on non-CM flights. But here is PAX information
    p1 = p1[(p1['carrier'] == 'CM')]

    p1['count'] = p1.groupby(['recloc','t_od'])['flt_num'].transform('count')  # THIS is NOT WORKING CORRECTLY!

    # If a route has only one leg, there are no feeders

    # Only keep the routes that have two legs (segments)
    count12 = p1.groupby('count').get_group(2)

    # count12_orig: feeder flights land at PTY
    # count12_dest: flights connected to depart from PTY
    count12_orig = count12.loc[count12['ptyorig'] == 0, :]
    count12_dest = count12.loc[count12['ptyorig'] == 1, :]

    # Calculate the number of arrivals at PTY per route (they should all be 1)
    count12_orig.loc[:,'count1'] = count12_orig.groupby(['recloc','t_od'])['miles'].transform('count')

    # Any group that has a count1 of 2, would imply two flights arriving at PTY, and therefore,
    # have no associated connecting flight (I am almost sure). So remove these rows
    ix = count12_orig[count12_orig['count1'] == 2].index
    count12_orig.drop(index=ix, axis=0, inplace=True)

    # Merge count12_orig and count12_dest with an outer join (keep all records of both)
    # These two dataframes have no overlap
    yyy = count12_orig.merge(count12_dest, how='outer',on=['recloc','t_od'], suffixes=["_f", "_nf"])

    # Only keep connections from 'CM' to 'CM' at PTY (only these can be found in FSU table)
    yyy = yyy[(yyy['carrier_f'] == 'CM') & (yyy['carrier_nf'] == 'CM')]

    ix = count12_orig[count12_orig['count1'] == 2].index
    xxx = count12_orig.drop(index=ix, axis=0)

    count12 = p1.groupby('count').get_group(2)

    count12_orig = xxx.copy()  # Already used!

    yyy = count12_orig.merge(count12_dest, how='outer',on=['recloc','t_od'], suffixes=["_f", "_nf"])

    # Remove nonCM carriers
    # This could reduce the number of feeders substantially since many are non-Copa international flights
    yyy = yyy[(yyy['carrier_f'] == 'CM') & (yyy['carrier_nf'] == 'CM')]

    # Sum pax_f
    # Make sure that each flights is uniquely identified. Otherwise the total_pax count will not be correct (it must be less than the size of an aircraft)

    cols_f  = ['dep_dtz_f',  'dep_tmz_f',  'od_f', 'flt_num_f']
    cols_nf = ['dep_dtz_nf', 'dep_tmz_nf', 'od_nf', 'flt_num_nf']
 
    # The reason these pax numbers are higher than plane capacity is because
    # on occasion, two planes with different flight numbers depart an airport
    # at exactly the same scheduled time on the same day and at the same time. 
    # No idea why this would ever happen!
    # Therefore, it is important to add 'flt_num_f' to cols_f 
    # and 'flt_num_nf' to cols-nf

    # Total number of passengers boarding one non-feeder flight from all feeders

    # Total number of passengers deplaning from a feeder and boarding another flight from PTY
    # total number of passengers cannot be larger than the size of the plan they come from!
    print("yyy.shape: ", yyy.shape)

    # columns index_f, index_nf are the rows in the original input file where the full record can be found

    cols = ['dep_dtz_nf','dep_tmz_nf','arr_dtz_nf','arr_tmz_nf','od_nf','flt_num_nf', 'dep_dtz_f','dep_tmz_f', 'cabin_nf','miles_nf','cabin_f','miles_f','arr_dtz_f','arr_tmz_f','od_f','flt_num_f','pax_f','pax_nf', 'carrier_f','carrier_nf','ampm_f','ampm_nf']


    # Create AM/PM column (PTY time)
    yyy['ampm_nf'] = time_2_AM_PM(yyy, 'dep_tmz_nf')
    yyy['ampm_f']  = time_2_AM_PM(yyy, 'dep_tmz_f')

    yyy[cols].sort_values(by=cols_f, axis=0).to_csv(output_fn, index=0)
    #yyy[cols].sort_values(by=cols_nf, axis=0).to_csv("flights_to_feeders.csv", index=0)

    return yyy

#----------------------------------------------------
# The parameter nbins is the number of bins
def plotFeederHist(dflist, binrange=(0.5,100.5), labels=('', ''), cols=('nb_feeders', 'nb_feeders'), \
                  title="Number of Feeders Histogram", alpha=(.5,.5)):
    """
    Plot a histogram of the number of feeders of two data frames.

    plotFeederHist(dflist, binrange=(0.5,100.5), labels=('', ''), cols=('nb_feeders', 'nb_feeders'),
                  title="Number of Feeders Histogram"):
    Arguments:
    - dflist : list of dataframes. Must contain the column: 'nb_feeders'
    - binrange : bins to display
    - cols : columns to plot for each histogram. Usually, it is the same column.
    - title: plot title with a default: "Number of Feeders Histogram".
    - alpha: transparency of each plot. Default: [0.5, 0.5]
    """
    feed1, feed2 = dflist
    nbins = int(binrange[1] - binrange[0])
    ax = sns.histplot(data=feed1, x=cols[0], hue=None, binwidth=1, binrange=(binrange[0],binrange[0]+nbins), color='red', alpha=alpha[0], label=labels[0])
    sns.histplot(data=feed2, x=cols[1], hue=None, binwidth=1, binrange=(binrange[0],binrange[0]+nbins), color='blue', alpha=alpha[1], label=labels[1])
    ax.set_xlim(-.5,binrange[1])
    plt.title(title, fontsize=20)
    plt.legend()
#---------------------------------------------------
def savePivotTables(df):
    """
    savePivotTables(df).

    Compute and save pivot tables to store departure and arrival dates and times of feeders, along with od_f and px_f.
    Return a list of pivot tables:  odf, paxf, depdtzf, deptmzf, arrdtzf, arrtmzf
        odf: feeder Origin-Destination
        paxf: feeder pax
        depdtzf: feeder departure date
        deptmzf: feeder departure time
        arrdtzf: feeder arrival date
        arrtmzf: feeder arrival time
    """
    depdtzf = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='dep_dtz_f', aggfunc=list).stack()
    print("depdtzf ...")
    deptmzf = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='dep_tmz_f', aggfunc=list).stack()
    print("deptmzf ...")
    arrdtzf = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='arr_dtz_f', aggfunc=list).stack()
    print("arrdtzf ...")
    arrtmzf = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='arr_tmz_f', aggfunc=list).stack()
    print("arrtmzf ...")
    odf     = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='od_f', aggfunc=list).stack()
    print("odf ...")
    paxf    = df.pivot_table(index=['dep_dtz_nf','dep_tmz_nf','od_nf'], columns='flt_num_f', values='pax_f', aggfunc=list).stack()
    print("paxf ...")
    depdtzf.reset_index().to_json("feeder_depdtzf.json")
    print("Wrote feeder_deptmzf.json")
    deptmzf.reset_index().to_json("feeder_deptmzf.json")
    print("Wrote feeder_deptmzf.json")
    arrdtzf.reset_index().to_json("feeder_arrdtzf.json")
    print("Wrote feeder_arrdtzf.json")
    arrtmzf.reset_index().to_json("feeder_arrtmzf.json")
    print("Wrote feeder_arrtmzf.json")
    paxf.reset_index().to_json("feeder_paxf.json")
    print("Wrote feeder_paxf.json")
    odf.reset_index().to_json("feeder_odf.json")
    print("Wrote feeder_odf.json")
    print("Wrote feeder_paxf.json")

    depdtzf.reset_index().to_csv("feeder_depdtzf.csv", index=0)
    print("Wrote feeder_deptmzf.csv")
    deptmzf.reset_index().to_csv("feeder_deptmzf.csv", index=0)
    print("Wrote feeder_deptmzf.csv")
    arrdtzf.reset_index().to_csv("feeder_arrdtzf.csv", index=0)
    print("Wrote feeder_arrdtzf.csv")
    arrtmzf.reset_index().to_csv("feeder_arrtmzf.csv", index=0)
    print("Wrote feeder_arrtmzf.csv")
    paxf.reset_index().to_csv("feeder_paxf.csv", index=0)
    print("Wrote feeder_paxf.csv")
    odf.reset_index().to_csv("feeder_odf.csv", index=0)
    print("Wrote feeder_odf.csv")

    return odf, paxf, depdtzf, deptmzf, arrdtzf, arrtmzf
#------------------------------------------------
def series_to_date_components(series, prefix="", suffix=""):
    """
    Decomposes a date column into day, week, dayofweek, month, year

        Parameters:
            series (DataSeries): A date column of a dataframe as a string
            prefix (string): prefix added to column names [default: ""]
            suffix (string): suffix added to column names [default: ""]

        Return (DataFrame): a dataframe with the day, week, dayofweek,
            month, year as columns
    """

    date = pd.to_datetime(series)
    month = date.dt.month
    year = date.dt.year
    day = date.dt.day
    week = date.dt.isocalendar().week
    dayofweek = date.dt.dayofweek
    titles = np.asarray(['d','w','dw','mo','y'])
    t = list(map(lambda x: (prefix+x+suffix), titles))
    df = pd.DataFrame({t[0]:day, t[1]:week, t[2]: dayofweek, t[3]:month, t[4]:year})
    return df
#------------------------------------------------
def series_to_time_components(series, prefix="", suffix=""):
    """
    Decomposes a date columns into hours/min

        Parameters:
            series (DataSeries): A date column of a dataframe as a string
            prefix (string): prefix added to column names [default: ""]
            suffix (string): suffix added to column names [default: ""]

         Return (DataFrame): a dataframe with the day, week, dayofweek,
            month, year as column
    """

    date = pd.to_datetime(series)
    hour = date.dt.hour
    minute = date.dt.minute
    titles = np.asarray(['h','mi'])
    t = list(map(lambda x: (prefix+x+suffix), titles))
    df = pd.DataFrame({t[0]:hour, t[1]:minute})
    return df

#---------------------------------------------------
def Zulu2PTY(datetime):
    """
    Convert from Zulu to PTY date-time.

    Parameters: 
        datetime [Series]: a series of date/times stored in nanoseconds integer format

    Return: 
        A series of date/times in nanoseconds integer format in Panama city

    If it is 1:00 pm Zulu, it is 6:00 pm PTY time (DTML)
    """

    # PTY time is earlier than Zulu time by 5 hours. However, data from the web suggests 
    # that PTY is 5 hours later than Zulu, and even 4 hours later according to one site. I have no idea what the truth is. 
    # According to Miguel, Panama is always 5 hours BEHIND Zulu

    return datetime - 5 * 3600 * 1000000000 # 5 hours difference # clutching at straws
#--------------------------------------------------
def per_interval(df1, kind='SCHED', bin_size=60, min_delay=None, max_delay=None, title="", ax=None, date=None, daterange=None, desired_timezone='Zulu'):
    """
    Plot the number of flights per hours computed from the input dataframes.

    Parameters:
        df1 (DataFrame) : Dataframe with all relevant data
        fsu_arr1 (DataFrame) : arrivals over "n" days
        bin_size (int) [60] : bin size in minutes (default: 60 min)
        kind (string) ['SCHED']: 'SCHED', 'ACTUAL', 'DELAY'
        min_delay (int) [None] : plot flights with delay > min_delay
        max_delay (int) [None] : plot flights with delay < max_delay
        title (string) [""] : plot title 
        ax (plt.Axis) [None] : axis for use in multiplot figures
        date (string) [None] : collect date on this date
        daterange (list or tuple) [None] : a date range over which to collect data
        desired_timezone (string) ['Zulu'] : Timezone used to collect the data within the date range

    Return:
        ax : axis object associated with the plot

    The input dataframe (df1) must contain the following columns, and is not modified during the call: 

    ['SCH_DEP_DTMZ','SCH_ARR_DTMZ','DEST_CD','ORIG_CD','IN_DTMZ','OUT_DTMZ','SCH_DEP_TMZ','SCH_ARR_TMZ']

    Only allow the following interval values (in min): 1, 5, 15, 30, 60

    Either the daterange or the date is None. Not both. 
    """

    bin_sizes = [1, 2, 3, 4, 5, 10, 15, 30, 60]
    tick_skips = [120, 60, 40, 30, 24, 12, 8, 4, 2]
    tick_skips = [x // 2 for x in tick_skips]
    
    cols = ['SCH_DEP_DTMZ','SCH_ARR_DTMZ','DEST_CD','ORIG_CD','IN_DTMZ','OUT_DTMZ','SCH_DEP_TMZ','SCH_ARR_TMZ','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES','OD','FLT_NUM','SCH_DEP_DTML_PTY','SCH_ARR_DTML_PTY']
    df = df1[cols].copy()

    for i,bin in enumerate(bin_sizes):
        if bin_size == bin:
            tick_skip = tick_skips[i]
            print("tick_skip= ", tick_skip)
            flag = 1

    if flag == 0:
        print("Non-valid bin size)")
        return

    if date == None and daterange == None:
        print("date and daterange cannot both be None!")
        return ax

    if date == None:
        min_date, max_date = daterange
    else:
        min_date = max_date = date

    if min_delay == None:
        min_delay = -9000
    if max_delay == None:
        max_delay =  9000 # (I code missing data as 9999)

    # Data is stored in Zulu time
    if desired_timezone == 'PTY':
        #print("Desired Timezone: ", desired_timezone)
        df['SCH_ARR'] = Zulu2PTY(df['SCH_ARR_DTMZ'])
        df['SCH_DEP'] = Zulu2PTY(df['SCH_DEP_DTMZ'])
        df['SCH_ARR_PTY'] = df['SCH_ARR_DTML_PTY'] # for testing
        df['SCH_DEP_PTY'] = df['SCH_DEP_DTML_PTY'] # for testing
        df['IN']  = Zulu2PTY(df['IN_DTMZ'])
        df['OUT'] = Zulu2PTY(df['OUT_DTMZ'])
    else:
        # Original data is in Zulu time
        df['SCH_ARR'] = df['SCH_ARR_DTMZ']
        df['SCH_DEP'] = df['SCH_DEP_DTMZ']
        df['IN']  = df['IN_DTMZ']
        df['OUT'] = df['OUT_DTMZ']


    # Extract records within the desired date range
    if kind == 'SCHED':
        # Scheduled departure and landing times
        arr_dates = pd.to_datetime(df['SCH_ARR']).dt.strftime("%Y-%m-%d")
        dep_dates = pd.to_datetime(df['SCH_DEP']).dt.strftime("%Y-%m-%d")
    else: 
        # Wheels leave or touch the ground
        arr_dates = pd.to_datetime(df['IN']).dt.strftime("%Y-%m-%d")
        dep_dates = pd.to_datetime(df['OUT']).dt.strftime("%Y-%m-%d")

    # Return the number of daily flights per time interval (bin_size)

    #### SOMETHING WRONG IN THIS SECTION =============================

    nb_days_arr = 1
    nb_days_dep = 1
    fsu_oneday_arr = df[
                        (arr_dates >= min_date) 
                      & (df['DEST_CD'] == 'PTY')  # Somehow this line is the issue
                      & (arr_dates <= max_date)
                      & (df['ARR_DELAY_MINUTES'] > min_delay)
                      & (df['ARR_DELAY_MINUTES'] < max_delay)
                      ].copy()
    fsu_oneday_dep = df[
                        (dep_dates >= min_date) 
                      & (df['ORIG_CD'] == 'PTY')
                      & (dep_dates <= max_date)
                      & (df['DEP_DELAY_MINUTES'] > min_delay)
                      & (df['DEP_DELAY_MINUTES'] < max_delay)
                      ].copy()

    #print("nb > max_delay: ", (df['DEP_DELAY_MINUTES'] < max_delay).sum())
    #print(min_delay, max_delay)
    #print(fsu_oneday_arr.shape, fsu_oneday_dep.shape)

    # Copy is necessary
    fsu_dep = fsu_oneday_dep.copy()
    fsu_arr = fsu_oneday_arr.copy()

    # Work with scheduled or actual times of departure
    if kind == 'SCHED':
        # Collect times of departure
        dep_tmz = pd.to_datetime(fsu_dep['SCH_DEP']) #.dt.time
        tmz = series_to_time_components(dep_tmz) 
    else:
        dep_tmz = pd.to_datetime(fsu_dep['OUT']) #.dt.time
        tmz = series_to_time_components(dep_tmz) 

    fsu_dep.loc[:,'min'] = (tmz['h']*60 + tmz['mi']) 
    fsu_dep.loc[:,'bin'] = fsu_dep['min'] // bin_size # Integer division. Returns bin number

    # Aggregate flights by bin
    fsu_dep.loc[:,'per_bin'] = fsu_dep.groupby('bin')['FLT_NUM'].transform('size')  / nb_days_dep

    # Work with scheduled or actual times of arrival
    if kind == 'SCHED':
        arr_tmz = pd.to_datetime(fsu_arr['SCH_ARR']) #.dt.time
        tmz = series_to_time_components(arr_tmz) 
    else:
        arr_tmz = pd.to_datetime(fsu_arr['IN']) #.dt.time
        tmz = series_to_time_components(arr_tmz) 

    fsu_arr.loc[:,'min'] = (tmz['h']*60 + tmz['mi']) #// 15 tmz['h'].copy()
    fsu_arr.loc[:,'bin'] = fsu_arr['min'] // bin_size # Integer division. Returns bin number
    # Aggregate flights by bin
    fsu_arr.loc[:,'per_bin'] = fsu_arr.groupby('bin')['FLT_NUM'].transform('size')  / nb_days_arr

    nb_bins_per_hour = 60 // bin_size
    # Mean is correct. It is a consequence of the transform() above

    sch_dep = series_to_time_components(fsu_dep['SCH_DEP'])
    sch_dep_pty = series_to_time_components(fsu_dep['SCH_DEP_DTML_PTY'])
    sch_arr = series_to_time_components(fsu_dep['SCH_ARR'])
    sch_arr_pty = series_to_time_components(fsu_dep['SCH_ARR_DTML_PTY'])

    tmz2 = series_to_time_components(df1['SCH_ARR_DTML_PTY'])

    fsu_dep = fsu_dep.groupby(['bin'])[['per_bin']].mean()
    fsu_arr = fsu_arr.groupby(['bin'])[['per_bin']].mean()

    # Fill in missing bins. Make sure there is an entry for every bin
    fsu_dep = fsu_dep.reindex(range(24*nb_bins_per_hour), fill_value=0.)
    fsu_arr = fsu_arr.reindex(range(24*nb_bins_per_hour), fill_value=0.)
    fsu_dep['dep-arr'] = 'dep'
    fsu_arr['dep-arr'] = 'arr'
    fsu = pd.concat([fsu_dep, fsu_arr], axis=0)

    # The copy is to avoid the slicing error from Pandas
    data = fsu[['per_bin','dep-arr']].reset_index('bin').copy()
    # mean and sum should give the same result
    # <<< is sum() correct?
    hours = data.groupby(['bin','dep-arr'])['per_bin'].sum().reset_index() 
    # -5 to convert from Zulu to Panama time

    ### NEED A GENERAL FORMULAS for ALL BIN SIZES
    #hours['h'] = ((hours['h'] - 5 + 24) % 24)   # Convert to Panama time

    # If I skip the translation in time, I get exactly the same plot, independent of timezone
    if desired_timezone == 'Zulu':
        print("***** CONVERT FROM Zulu to Panama time in the same day, *** NOT SMART SINCE DAY MIGHT CHANGE")
        #hours.loc[:,'bin'] = ((hours['bin'] - 5*nb_bins_per_hour + 24*nb_bins_per_hour) % (24*nb_bins_per_hour)) / nb_bins_per_hour  
        pass

    hours.sort_values('bin', inplace=True)

    deparr = hours.groupby('dep-arr').sum()
    print(deparr['per_bin'])
    print(f"total flights (dep+arr): {hours['per_bin'].sum()}")

    if date == None:
        title += f", {kind}, ({daterange})"
    else:
        title += f", {kind}, ({date})"
    title += f"\n delay: ({min_delay,max_delay})"
    ax.set_title(title, fontsize=18)

    # HERE IS THE PROBLEM: I must fill in the missing x-data. Use reset_index()
    # The bars are always uniformly-spaced. That is the issue.

    # Try and remove warning error generated by sns.barplot
    hours = hours.copy()

    sns.barplot(data=hours,x='bin',y='per_bin',hue='dep-arr', ax=ax)
    plt.setp(ax.patches, linewidth=0)  # remove stroke from bars

    #print("==> hours shape,min,max: ", hours.shape, hours['bin'].min(), hours['bin'].max())

    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()   # How to skip labels?
    labels = [str(x/tick_skip) for x in ticks]

    plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(ticks[0::tick_skip]);  # suppress output
    ax.set_xticklabels(labels[0::tick_skip], fontsize=8);  # suppress output
    ax.set_xlabel(f"Time of day (hours), binsize={bin_size}min ({desired_timezone} time)")
    ax.grid(True)

    #plt.show()
    return ax
#----------------------------------------------------
def delayByOD(df, delay_range=(None, None), filenm='od_arrival_delays_med,mean,std.csv'):
    """
    Compute normal distribution fits to the arrival and delay distributions, by OD, across the full time period.

    Parameters: 
    df (DataFrame) : input dataframe. Must contain the columns: ARR_DELAY_MINUTES, DEP_DELAY_MINUTES, OD)
    delay_range (tuple) [(None, None)]: minimum and maximum delay to consider in the calculation. If None, use a very large value set to -100000 or +100,000 for lower and upper limit, respectively.

    Return: 
    A tuple of DataFrames. One for arrivals and one for departures. 

    Depending on the range of delays, the number of OD pairs in both files might be different.
    """
    min_delay, max_delay = delay_range
    fn_suffix = f"_delayrange={str(min_delay)}-{str(max_delay)}.csv"

    if min_delay == None:
        min_delay = -100000
    if max_delay == None:
        max_delay = 100000

    print(min_delay, max_delay)

    arr_d = df[df['ARR_DELAY_MINUTES'].between(min_delay,max_delay)].rename(columns={'ARR_DELAY_MINUTES':'ARR_D'})
    dep_d = df[df['DEP_DELAY_MINUTES'].between(min_delay,max_delay)].rename(columns={'DEP_DELAY_MINUTES':'DEP_D'})

    arr_od_med = arr_d.groupby('OD').median().rename(columns={'ARR_D':'ARR_D_MED'})
    dep_od_med = dep_d.groupby('OD').median().rename(columns={'DEP_D':'DEP_D_MED'})

    arr_od_mean = arr_d.groupby('OD').mean().rename(columns={'ARR_D':'ARR_D_MEAN'})
    dep_od_mean = dep_d.groupby('OD').mean().rename(columns={'DEP_D':'DEP_D_MEAN'})

    arr_od_std = arr_d.groupby('OD').std().rename(columns={'ARR_D':'ARR_D_STD'})
    dep_od_std = dep_d.groupby('OD').std().rename(columns={'DEP_D':'DEP_D_STD'})

    arr_od = pd.concat([arr_od_med['ARR_D_MED'], arr_od_mean['ARR_D_MEAN'], arr_od_std['ARR_D_STD']], axis=1)
    dep_od = pd.concat([dep_od_med['DEP_D_MED'], dep_od_mean['DEP_D_MEAN'], dep_od_std['DEP_D_STD']], axis=1)

    filenm = "od_dep_arr_delays" + fn_suffix
    arr_od.reset_index().to_csv(filenm, index=0, float_format='%10.3e')
    dep_od.reset_index().to_csv(filenm, index=0, float_format='%10.3e')

    return arr_od, dep_od

#----------------------------------------------------------
def time_2_AM_PM(df, time_col):
    """
    Convert DTMZ to AM/PM at Panama City (PTY)
    """
    nano = pd.to_datetime(df[time_col]).astype('int')
    nano -= 5 * 3600 * 1000000000
    hour = pd.to_datetime(nano).dt.hour
    def ampm(x):
        if x > 12: return 'PM'
        return 'AM'
    return hour.apply(ampm)
#----------------------------------------------------------------------
def createIdPair(df1, dest_col, cols, suffixes = ['_f','_nf']):
    """

    Append flight IDs to the input bookings table.

    Parameters: 

       df1 (pandas.DataFrame) : input dataframe to enhance with flight identifiers
       dest_col (string) : base name of the identifier columns
       cols (list(string)) : columns from which to form the identifier
       suffixes (list(string)) : suffixes to add to the base identifier column header

    Return: 
       A dataframe with two additional identifier columns. 

    The original input remains intact. The number of rows in the returned dataframe
    is identical to the number of rows in the input dataframe.

    The base name has the suffixes added to it according to the parameter `suffixes`.

    """

    df = df1.copy()

    if (len(suffixes) != 2):
        print("length of suffixes must be 2")
        return -1

    for c in cols:
        print("1 col= ", c)
        if type(df[c+suffixes[0]].values[0]) != str:
            df[c+suffixes[0]] = df[c+suffixes[0]].astype('str')
        if type(df[c+suffixes[1]].values[0]) != str:
            df[c+suffixes[1]] = df[c+suffixes[1]].astype('str')

    col = dest_col+suffixes[0]
    df[col] = df[cols[0]+suffixes[0]]
    for i in range(1,len(cols)):
        df[col] += df[cols[i]+suffixes[0]]

    col = dest_col+suffixes[1]
    df[col] = df[cols[0]+suffixes[1]]
    for i in range(1,len(cols)):
        df[col] += df[cols[i]+suffixes[1]]
    return df

#---------------------------------------------------------------------
def createUniqueIdPairList(df, idcol, cols, suffixes=['_f','_nf']):
    """

    Return a column of unique (feeder/non-feeder) flight identifier pairs.

    Parameters:

       df (pandas.DataFrame) : input dataframe 
       idcol (string) : name of the identifier column
       cols (list(string)) : columns from which to form the identifier
       suffixes (list(string)) : suffixes to add to the base identifier column header

    Return:
       A dataframe with two additional identifier columns.

    The returned list has less rows than the input list since a feeder flight often has 
    multiple connecting passengers conneting to the same flight, and who booked their 
    reservations independently. 

    """
    idcols = [idcol + x for x in suffixes]
    #idcols = [idcol+suffixes[0], idcol+suffixes[1]]
    dfids = createIdPair(df, idcol, cols, suffixes)[idcols].drop_duplicates()
    print(dfids.shape)
    return dfids


#----------------------------------------------------------------------
def createId(df1, dest_col, cols):
    """

    Append flight IDs to the input table (usually the FSU table).

    Parameters:

       df1 (pandas.DataFrame) : input dataframe to enhance with flight identifiers
       dest_col (string) base name of the identifier column
       cols (list(string)) : columns from which to form the identifier

    Return:
       A dataframe with one additional identifier column.

    The original input remains intact. The number of rows in the returned dataframe
    is identical to the number of rows in the input dataframe.

    """
    df = df1.copy()
    df[dest_col] = df[cols[0]]
    for i in range(1,len(cols)):
        try: 
            df[dest_col] += df[cols[i]]
        except:
            # In case the colum is integer or float type
            df[dest_col] += str(df[cols[i]])
    return df

def createUniqueIdList(df, idcol, cols):
    """
    Starting from the input dataframe (df), return a list of unique flight identifiers

    Parameters: 

       df (pandas.DataFrame) : input dataframe 
       idcol (string) : name of the flight Id column
       cols (list(string)) : columns from which to derive the flight identifier.

    Return: 
        A Data Series with a list of unique flight identifiers.

    """
    dfids = createId(df, idcol, cols)[idcol].drop_duplicates()
    print(dfids.shape)
    return dfids

#----------------------------------------------------------------------
def createIdentifiers(fsu_1, dff_1, idcol, fsu_cols, dff_cols):
    """
    Create ID lists for `fsu` and `dff` dataframes with duplicates removed

    Parameters:
    fsu (pandas.DataFrame) : FSU dataframe
    dff (pandas.DataFrame) : Bookings dataframe
    idcol (string) : name of the identifier column
    fsu_cols (list(string)) : list of column headers out of which the FSU identifier is created
    dff_cols (list(string)) : list of column headers out of which the Bookings identifier is created

    Return : Two dataframes stored in a tuple (df, dff): 
       df  (pandas.DataFrame) : single column `id` of unique identifiers
       dff (pandas.DataFrame) : two columns [`id_f`,`id_f`] of identifiers from the Bookings table

    Each pair (`id_f`,`id_nf`) is unique, and corresponds to a feeder-non-feeder pair.. 
    """

    # Copy for safety. I feel that the argument is passed by reference
    fsu = fsu_1.copy()
    dff = dff_1.copy()

    # Make sure that all the columns are strings
    # Updates to the data frames are not propagated to the context of the calling function

    for c in fsu_cols:
        if type(fsu[c].values[0]) != str:
            fsu[c] = fsu[c].astype('str')

    for c in dff_cols:
        cf = c + '_f'
        if type(dff[cf].values[0]) != str:
            dff[cf] = dff[cf].astype('str')
        cnf = c + '_nf'
        if type(dff[cnf].values[0]) != str:
            dff[cnf] = dff[cnf].astype('str')


    fsu1 = createUniqueIdList(fsu, idcol, fsu_cols)
    dff1 = createUniqueIdPairList(dff, idcol, dff_cols)
    print(f"dff1: {dff1.shape}, fsu1: {fsu1.shape}")
    return (fsu1, dff1)

#----------------------------------------------------------------------
def noRecordsFound():
    print("No records found")
    return pd.DataFrame()

#---------------------------------------------------------------------
def fsuRecords(df_ids, ids, id_col='id', cols=None):
    """
    Starting with FSU table, return the rows that contain the records in `ids`

    Arguments: 
        df_ids (DataFrame): fsu dataframe with an 'id' column appended
        ids: Dataframe with a column 'id' (default) 
        id_col: Name of 'id' column
        cols: list of columns returned. Default: None.

    Return: 
        A DataFrame that is the subset of 'df_ids' with the ids in 'ids'. 
        If there is a processing error, return noRecordsFound(), which itself
        prints an error message and returns an empty dataframe.

    """
    if type(df_ids) != pd.DataFrame:
        print("df_ids must be a DataFrame")
        return None

    if type(ids) != pd.core.series.Series:
        print("ids must be a Data Series")
        return None

    if cols == None:
        try:
            dd = df_ids.set_index(id_col)
            df_out = df_ids.set_index(id_col).loc[ids, :].reset_index()
        except:
            return noRecordsFound()
    else:
        try:
            d_out = df_ids.set_index(ids).loc[ids, cols].reset_index()
        except:
            return noRecordsFound()

    if df_out.shape[0] != ids.shape[0]:
        print("Input/Output number of rows mismatch: ", ids.shape[0], df_out.shape[0])

    return df_out

#-------------------------------------------------------------------------
def bookingRecords(df_ids, ids, cols=None):
    if type(ids) != pd.core.series.Series:
        print("ids must be a Data Series")
        return None

    id_col = ids.name  # ids must be a Data Series
    if cols == None:
        try:
            # Very expensive ids.shape[0] is very high (like 400,000)
            df_out = df_ids.set_index(id_col)  # Very Fast
            df_out = df_out.loc[ids, :].reset_index()  # The slow line
        except:
            return noRecordsFound()
    else:
        try:
            df_out = df_ids.set_index(id_col).loc[ids, cols].reset_index()
        except:
            return noRecordsFound()

    print("5")
    if df_out.shape[0] != ids.shape[0]:
        print("Input/Output number of rows mismatch: ", ids.shape[0], df_out.shape[0])

    return df_out

#----------------------------------------------------------------------
def aggregateBookings(dff_ids, dff_id_cols, save_data=False, output="bookings", out_suffix=("_f","_nf"), dest_dir="."):
    """

    Aggregate the Pax on all feeder-outbound pairs. 

    Parameters: 

        dff_ids (pandas.DataFrame) : bookings table, including feeder-non-feeder pairs and flight identifiers.
        dff_id_cols (list(string)) : columns to use to aggregate data (does not include _f and _nf).
        save_data (bool) : whether to save the data or not.
        output (string) : output file name.
        outsuffixes (list(string)), default ("_f","_nf") : suffixes for the output file name .
        dest_dir, default (".") : relative output folder path

        return (tuple): list of feeder flights and outbound flights. The feeder table contains the feeder ids, 
        and the outbound flight table includes the outbound flight ids. 
    """

    dff_id_cols_f = list(map(lambda x: x+'_f', dff_id_cols)) + ['id_f']
    dff_id_cols_nf = list(map(lambda x: x+'_nf', dff_id_cols)) + ['id_nf']
    dff_all_cols = dff_id_cols_f + dff_id_cols_nf

    # This excluded the ids
    dff_ids_numeric_cols = dff_ids.select_dtypes(include='number').columns
    dff_ids_cols = list(dff_ids_numeric_cols) + ['id_f','id_nf']
    print("Numerical columns in dff_ids_cols + ['id_f','id_nf']: ", dff_ids_cols)

    # Group by nf+f combination, so that there is only a single row for each _f + _nf pair
    # All the columns are summed up.
    dff_ids_g = dff_ids.groupby(dff_id_cols_nf + dff_id_cols_f)[dff_ids_cols].sum()
    print("dff_ids_g.columns: ", dff_ids_g.columns)
    dff_ids_g = dff_ids_g.reset_index()
    print("dff_ids_g.reset_index().columns: ", dff_ids_g.columns)


    s1 = dff_ids.groupby('id_f')['id_nf'].size().max()
    s2 = dff_ids.groupby('id_nf')['id_f'].size().max()
    print("dff_ids: s1,s2= ", s1, s2)
    s1 = dff_ids_g.groupby('id_f')['id_nf'].size().max()
    s2 = dff_ids_g.groupby('id_nf')['id_f'].size().max()
    print("dff_ids_g: s1,s2= ", s1, s2)
    print("dff_ids.columns: ", dff_ids_g.columns)


    print("dff_ids.columns: ", dff_ids.columns)
    #print("dff_ids: ", dff_ids.shape)
    #print("dff_id_cols_nf: ", dff_id_cols_nf)
    dff_nf = dff_ids.groupby(dff_id_cols_nf)[dff_ids_numeric_cols].sum().reset_index()
    #print("dff_nf: ", dff_nf.shape)

    # Aggregate by _f flights. Each row contains the total pax debarking from the _f flight and connecting to another flight at PTY.

    dff_f = dff_ids.groupby(dff_id_cols_f)[dff_ids_cols].sum().reset_index()
    #print(dff_f.shape)

    # Aggregate by _f flights. Each row contains the total pax debarking from the _f flight and connecting to another flight at PTY.

    # dff_ids: function argument
    # dff_ids_g: computed in this function. 

    if save_data:
        dff_f.to_csv(dest_dir+output+out_suffix[0]+".csv.gz", index=0)
        dff_nf.to_csv(dest_dir+output+out_suffix[1]+".csv.gz", index=0)
        dff_ids.to_csv(dest_dir+"_ids+pax.csv.gz", index=0)
        dff_ids_g.to_csv(dest_dir+output+"_ids_g+pax.csv.gz", index=0)

    # In practice, use these two files to search for feeder and non-feeder ids
    return dff_f, dff_nf, dff_ids_g[['id_f', 'id_nf', 'pax_f']]

#---------------------------------------------------------------------
def new_aggregateBookings(dff_ids, dff_id_cols, save_data=False, output="new_bookings", out_suffix=("_f","_nf"), dest_dir="."):
    """
    Applies to new FSU table
    Aggregate the Pax on all feeder-outbound pairs. 

    Parameters: 

        dff_ids (pandas.DataFrame) : bookings table, including feeder-non-feeder pairs and flight identifiers.
        dff_id_cols (list(string)) : columns to use to aggregate data (does not include _f and _nf).
        save_data (bool) : whether to save the data or not.
        output (string) : output file name.
        outsuffixes (list(string)), default ("_f","_nf") : suffixes for the output file name .
        dest_dir, default (".") : relative output folder path

        return (tuple): list of feeder flights and outbound flights. The feeder table contains the feeder ids, 
        and the outbound flight table includes the outbound flight ids. 
    """

    dff_id_cols_f = list(map(lambda x: x+'_f', dff_id_cols)) + ['id_f']
    dff_id_cols_nf = list(map(lambda x: x+'_nf', dff_id_cols)) + ['id_nf']
    dff_all_cols = dff_id_cols_f + dff_id_cols_nf

    # This excluded the ids
    dff_ids_numeric_cols = dff_ids.select_dtypes(include='number').columns
    dff_ids_cols = list(dff_ids_numeric_cols) + ['id_f','id_nf']

    dff_ids_g = dff_ids.groupby(dff_id_cols_nf + dff_id_cols_f)[dff_ids_cols].sum().reset_index()
    #print(dff_ids.shape, dff_ids_g.shape)
    #display(dff_ids_g.head())

    print("dff_ids.columns: ", dff_ids.columns)
    #print("dff_ids: ", dff_ids.shape)
    #print("dff_id_cols_nf: ", dff_id_cols_nf)
    dff_nf = dff_ids.groupby(dff_id_cols_nf)[dff_ids_numeric_cols].sum().reset_index()
    #print("dff_nf: ", dff_nf.shape)

    # Aggregate by _f flights. Each row contains the total pax debarking from the _f flight and connecting to another flight at PTY.

    dff_f = dff_ids.groupby(dff_id_cols_f)[dff_ids_cols].sum().reset_index()
    #print(dff_f.shape)

    # Aggregate by _f flights. Each row contains the total pax debarking from the _f flight and connecting to another flight at PTY.

    if save_data:
        dff_f.to_csv(dest_dir+output+out_suffix[0]+".csv.gz", index=0)
        dff_nf.to_csv(dest_dir+output+out_suffix[1]+".csv.gz", index=0)
        dff_ids.to_csv(dest_dir+"_ids+pax.csv.gz", index=0)
        dff_ids_g.to_csv(dest_dir+output+"_ids_g+pax.csv.gz", index=0)

    # In practice, use these two files to search for feeder and non-feeder ids
    return dff_f, dff_nf, dff_ids_g[['id_f', 'id_nf', 'pax_f']]

#---------------------------------------------------------------------
#---------------------------------------------------------------------
def createMergeIdList(fsu_id, dff_id): 
    """
    Create a dataframe with two columns ('id_f', 'id_nf') that pair up feeders with non-feeders
    """
    d_merge = dff_id.merge(fsu_id.to_frame(), how='inner', 
              left_on='id_f', right_on='id')[['id_f','id_nf']]
    d_merge1 = d_merge.merge(fsu_id.to_frame(), how='inner', 
              left_on='id_nf', right_on='id')[['id_f','id_nf']]
    merge2 = d_merge1.drop_duplicates() # there should be no duplicates
    print("fsu_id.shape, dff_id.shape, d_merge.shape, d_merge1.shape, merge2.shape")
    print(fsu_id.shape, dff_id.shape, d_merge.shape, d_merge1.shape, merge2.shape)
    return d_merge1

#---------------------------------------------------------------------
def mergeFSUBookings(fsu_ids, dff_ids, ids, cols_fsu=None, cols_dff=None):
    """
    # WORK IN PROGRESS. NOT WORKING

    Create a merged file with the results from fsuRecords and bookingRecords

    Parameters
    fsu_ids (DataFrame) : FSU table with a flight identifier column labeled as `id`.
    dff_ids (DataFrame) : Bookings table with a flight identifier columns labeled as `id_f` and `id_nf`.
    ids (List of string) : List of identifiers
    cols_fsu (List of string) : Column names from the fsu table to keep in merged result. If None, keep all columns.
    cols_dff (List of string) : Column names from the bookingss table to keep in merged result. If None, keep all columns.
    fsu: Bookings table.

    Return : the merged dataframe

    Starting from a set of id_f/id_nf pairs stored in `ids`, return a single table with all the
    """

    fsu_f  = fsuRecords(fsu_ids, f['id_f'])
    fsu_nf = fsuRecords(fsu_ids, f['id_nf'])
    dff_f  = bookingRecords(dff_ids, f['id_nf'])
    dff_nf = bookingRecords(dff_ids, f['id_nf'])
    print(fsu_f.shape, fsu_nf.shape, dff_f.shape, dff_nf.shape)

    return None

#-------------------------------------------------------------
def findFeederIds(df, outbound_id):
    return df[df['id_nf'] == outbound_id].copy()['id_f']

#-------------------------------------------------------------
#def findNonFeederIds(df, feeder_id):
def findOutboundIds(df, feeder_id):
    return df[df['id_f'] == feeder_id].copy()['id_nf']

#-------------------------------------------------------------
def constructOutbounds(df, network, nodes, pairs, feeder_ids, max_level=2, level=0):
    # Possibly create a dictionary rather than a list for faster access
    # Starting with a list of feeder IDs (`feeder_ids`), construct all the outbound flights connecting to it and assign level `level` to these nodes.
    outbounds = []
    levels = []

    for feeder_id in feeder_ids:
        outs = findOutboundIds(df, feeder_id).values
        for out in outs:
                outbounds.append(out)
                levels.append(level)
                nodes.append((out, level))
                pairs.append((feeder_id, out, level))
                #network[feeder_id] = out_list  # I assume no repetition
    return outbounds, levels

#--------------------------------------------------
def constructFeeders(df, network, nodes, pairs, outbound_ids, max_level=2, level=0):
    feeders = []
    levels = []

    # For each outbound flight, identify the return flight to PTY
    # I assume that tails only change in PTY and not at other airports. But that might be the case. [


    for outbound_id in outbound_ids:
        #print("new oubound: ", outbound_id)
        feeds = findFeederIds(df, outbound_id).values
        #out_list = []
        for feed_id in feeds:
                feeders.append(feed_id)
                levels.append(level) # inefficient
                nodes.append((feed_id, level))
                pairs.append((outbound_id, feed_id, level))
                #out_list.append(out)  # captures all pairs, even with interactions
                #network[feeder_id] = out_list  # I assume no repetition
    return feeders, levels
#--------------------------------------------------

def minimizeIdList(fsu_ids, id_list):
    """
    Return the common records between fsu_ids and id_list: ret_fsu_ids, ret_id_list. 
    Remove from fsu_ids, all rows that do not not exist in id_list

    Parameters: 
        fsu_ids, (DataFrame) :
        id_list, (DataFrame) :

    Return:
        ret_fsu_ids, (DataFrame) :
        ret_id_list, (DataFrame) :
    """

    # Collect all the id's in id_list
    ids = list(set(id_list['id_f'].tolist() + id_list['id_nf'].tolist()))

    fsu_ids_ = fsu_ids['id'].tolist()
    print("fsu_ids_: ", len(fsu_ids_))
    fsu_ids_ = list(set(fsu_ids_))
    print("fsu_ids_: ", len(fsu_ids_))

    print("ids: ", len(ids))

    # Get the IDs common to ids and fsu_ids
    ids_inters = set(fsu_ids_).intersection(ids)
    print("len(inters): ", len(ids_inters))

    # Reduce the size of id_list so that all ids in the list are contained in ids_inters

    ret_fsu_ids = fsu_ids.set_index('id').loc[ids_inters,:]

    ret_id_list = pd.DataFrame()

    return ret_fsu_ids, ret_id_list

#----------------------------------------------------------------
def constructFeederOutboundPairs(fsu_ids, min_delay, dep_date, id_list, nodes, pairs):
    """
    Given `id_list1`, a list of `(id_f,id_nf)` pairs, reduce the list to only include ids included
    in the FSU list. 
    """

    feeders = []
    levels = []
    cols = ['id','SCH_DEP_DTZ','SCH_DEP_TMZ','TAIL','OD','ARR_DELAY_MINUTES','DEP_DELAY_MINUTES','IN_DTMZ','OUT_DTMZ','OFF_DTMZ','ON_DTMZ']

    # find id_list in fsu_ids
    fsu_min = fsu_ids[fsu_ids['SCH_DEP_DTZ'] == dep_date][cols]
    print("fsu_min: ", fsu_min.shape)
    # no change to the list in the calling function. So not really passed by ref
    id_list = id_list[id_list['id_f'].str[0:10] == dep_date]

    # Only keep flights with departure delays greater than 15 min
    fsu_min = fsu_min[fsu_min['DEP_DELAY_MINUTES'] > min_delay]
    print("fsu_min: ", fsu_min.shape)

    # merge id_list with FSU database. Requires two merges, since both feeders
    # and nonfeeders must by in the FSU table

    d_merge = id_list.merge(fsu_min, how='inner', 
              left_on='id_f', right_on='id').drop('id_f', axis=1).drop(['id_nf'],axis=1)
    print("merge: delay < 0: ", d_merge[d_merge['DEP_DELAY_MINUTES'] < 0]) #empty

    d_merge1 = id_list.merge(fsu_min, how='inner', 
              left_on='id_nf', right_on='id', suffixes=['_f','_nf']).drop('id_f', axis=1) 
    d_merge3 = pd.concat([d_merge, d_merge1], axis=0)

    print("d_merge3: ", d_merge3.shape)
    d_merge3 = d_merge3.drop_duplicates()
    print("==> d_merge3: ", d_merge3.shape, d_merge3.columns)


    print("merge1: delay < 0: ", d_merge1[d_merge1['DEP_DELAY_MINUTES'] < 0]) #empty


    merge2 = d_merge1.drop_duplicates() # there should be no duplicates
    print("merge2: delay < 0: ", merge2[merge2['DEP_DELAY_MINUTES'] < 0]) #empty
    print("merge2 shape: ", merge2.shape)
    #print(merge2['DEP_DELAY_MINUTES'])

    # Zip requires lists as arguments
    s1 = merge2['id_f'].values.tolist()
    s2 = merge2['id_nf'].values.tolist()
    merge2['edges'] = [(e1,e2) for e1,e2 in zip(s1,s2)]

    # The problem is that some of the merge columns do not correspond to 'id_nf' or 'id_f' (do not know which one)
    edge_ids = merge2['edges'].tolist()
    # Remove duplicates
    edge_ids = list(set(edge_ids))

    s = s1 + s2

    # all edges. Do not need metadata for the edges
    print("merge2: ", merge2.shape)

    # Stack id_f and id_nf (using pd.melt?) and remove duplicates
    # What is not clear is what the FSU records correspond to? id_f or id_nf? 
    # SOMETHING STILL WRONG

    nodes = d_merge3['ids'].values
    print("len(node_ids)= ", len(node_ids))  # TOO MANY. WHY? 

    pairs.extend(edge_ids)

    edge_ids = pd.DataFrame({'e1':merge2['id_f'], 'e2':merge2['id_nf']})
    node_ids = pd.DataFrame(node_ids, columns=['id'])

    return node_ids, edge_ids
#----------------------------------------------------------------
def difference(list1, list2):
    """
    Compute and return the elements in list1 not in list2 as a list.
    """
    return list(set(list1).difference(set(list2)))

#-------------------------------------------------------------
def intersection(list1, list2):
    """
    Compute and return the elements common to list1 and list2.
    
    `list1` and `list2` can be sets, lists, or pandas.Series.
    """
    return list(set(list1).intersection(set(list2)))

#-------------------------------------------------------------
def uniqueIds(fsu_ids, id_list):
    """
    Parameter:
        fsu_ids: list of ids in the fsu table
        id_list: list of ids pairs related to booking

    Return:
        fsu_valid_df: fsu_ids restricted to fsu_valid_ids (see next return value)
        fsu_valid_ids (pd.Series): a list of node ids. Each node id can be found among the feeders and outbound (nf) IDs in id_list.
        id_list_y (pd.DataFrame): pairs of feeders/outbounds such that both elements are contained in fsu_valid_ids.
    """
    ids_u = fsu_ids['id'].drop_duplicates()

    list_ids_u = pd.concat([id_list['id_f'],id_list['id_nf']], axis=0).drop_duplicates()

    fsu_valid_ids = intersection(list_ids_u, ids_u)

    # Use this in visualization notebook
    #attr = fsu_ids.set_index('id').loc[fsu_valid_ids, :].to_dict('id')

    fsu_valid_df = fsu_ids.set_index('id').loc[fsu_valid_ids, :].reset_index()

    x = intersection(ids_u, id_list['id_f'])   # Only feeders
    y = intersection(ids_u, id_list['id_nf'])  # Only nonfeeders

    id_list_x = id_list.set_index('id_f').loc[x, :].reset_index()
    id_list_y = id_list_x.set_index('id_nf').loc[y, :].reset_index()

    print('id_list: ', id_list.shape)
    print('id_list_x: ', id_list_x.shape)
    print('id_list_y: ', id_list_y.shape)

    return fsu_valid_df, fsu_valid_ids, id_list_y

#----------------------------------------------------------
def nbNaN(series):
    return series.isnull().values.sum()

#---------------------------------------------------------
def listNbNans(df):
# Nb of NaN in each column
    for col in df.columns:
        nb_nan = nbNaN(df[col])
        if nb_nan > 0:
            print(f"{col}: {nb_nan}")

#---------------------------------------------------------
def ID2DF(series):
    """
    Given a series of IDs, extract the OD

    Parameters:
        series: A Pandas Series or a List

    Return: 
        Return a dataframe with all the information extracted
    """

    orig = series.str[10:13]
    dest = series.str[13:16]
    date = series.str[0:10]
    od = series.str[10:16]
    hour_DTZ = series.str[16:18]
    minute = series.str[19:21]
    flt_num = series.str[21:]

    return pd.DataFrame({'ORIG':orig, 'DEST':dest, 'OD':od,  \
                        'hour_DTZ':hour_DTZ, 'min':minute, 'FLT_NUM':flt_num})

#---------------------------------------------------------
def findCity(df, city='MIA'):
    try:
        records = df[df['OD'].str.contains(city)]
    except:
        try:
            records = df[df['od_f'].str.contains(city)]
        except:
            records = df[df['od_nf'].str.contains(city)]
    return records
#---------------------------------------------------------
# streamlit cache
#@st.cache(suppress_st_warning=True)
# Error with cache when running in Jupyter-lab. Encountered an unashable object (a function)
def handleCity(city, choice_ix, id_list, fsu, bookings_f, feeders, is_print=True, delay=45):
    # Need to increase efficiency
    city_inbounds = findCity(bookings_f, city)
    #st.write("city_inbounds: ")
    #st.write(f"City: {city}")
    #st.write("city_inbounds: ")
    #st.write(city_inbounds)
    #st.write(choice_ix, id_list.shape, fsu.shape, bookings_f.shape, feeders.shape)

    if choice_ix == 'all':
        min_ix = 0
        max_ix = city_inbounds.shape[0]
    else:
        min_ix = choice_ix
        max_ix = choice_ix+1

    #st.write(f"min_ix: , {min_ix}")
    for which_ix in range(min_ix, max_ix):

        try:
            inbound = city_inbounds.iloc[which_ix].id_f
            #st.write(f"inbound: {inbound}")
            #st.write(f"fsu.shape: {fsu.shape}")
            print("inbound= ", inbound)
            fsu_inbound  = fsu.set_index('id').loc[inbound]
            #st.write(f"fsu_inbound: {fsu_inbound}")
        except:
            #st.write("first except")
            #print("first except")
            continue

        try:
            fsu_inbound  = fsu.set_index('id').loc[inbound]
            #st.write(f"fsu_inbound: ...")
            #st.write(f"{fsu_inbound}")
            #print(f"fsu_inbound: ...")
            #print(f"{fsu_inbound}")
        except:
            # Sometimes the key "inbound" is not found
            st.write("HandleCity: second exception")
            print("HandleCity: second exception")
            continue

        inbound_arr_delay = fsu_inbound.ARR_DELAY_MINUTES
        
        # if the delay is negative, the plane arrived early, and the 
        # passengers have time to connect
        if inbound_arr_delay < 0:
            #st.write("arr_delay < 0")
            #print("arr_delay < 0")
            #continue # <<< ADD THIS BACK?
            pass

        outbounds = findOutboundIds(id_list, inbound).to_frame()
        outbounds['id_f'] = inbound

        #st.write("feeders: ", feeders.shape)  # <<<< THE DIFFERENCE
        #print("feeders: ", feeders.shape)

        # Create a unique id that combines inbound (feeder) and outbound flights
        # This will allow me to merge two files with feeder/non-feeder columns
        feeders_1 = feeders[feeders['id_f'] == inbound]
        feeders_1['id_f_nf'] = feeders_1['id_f'] + feeders_1['id_nf']

        # extract these outgoings from the FSU database
        # if outbounds has ids not in fsu, this approach will not work
        fsu_outbound = pd.merge(fsu, outbounds, how='inner', left_on='id', right_on='id_nf')
        fsu_outbound['id_f_nf'] = fsu_outbound['id_f'] + fsu_outbound['id_nf']

        #st.write("fsu_outbound: ", fsu_outbound.shape)
        #st.write("feeders_1: ", feeders_1.shape)
        #print("fsu_outbound: ", fsu_outbound.shape)
        #print("feeders_1: ", feeders_1.shape)

        fsu_pax      = pd.merge(fsu_outbound, feeders_1, how='inner', left_on='id_f_nf', right_on='id_f_nf')
        fsu_pax.drop_duplicates(inplace=True)
        #print("fsu_pax.shape: ", fsu_pax.shape)

        # <<<<<< FIX THIS ERROR >>>>>>>
        #st.write(f"fsu_pax.shape:  {fsu_pax.shape}")   ### (0,99) on streamlit, (nonzero, 99) on Jupyter-lab

        #st.write("2")


        #print(fsu_pax.columns)
        #print(fsu_pax[['OD','id']])  # The id is non-feeder

        # Compute connection time (inbound.IN - outbound.sch_dep)
        #display(fsu_pax)
        available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().IN_DTMZ) / 1e9 / 60
        planned   = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().SCH_ARR_DTMZ) / 1e9 / 60
        delta = planned - available   # = IN - SCH_ARR
        dep_delay = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60
        #print('dep_delay= ', dep_delay.shape, delta.shape, planned.shape, available.shape)

        if is_print:
            #st.write(f"is_print: {is_print}")
            arr_delay = (fsu_pax.IN_DTMZ  - fsu_pax.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
            #print("fsu_pax: ", fsu_pax.shape)

            printed_header = False
            zipped = zip(planned, available, delta, arr_delay, dep_delay, fsu_pax.SCH_DEP_TMZ, fsu_pax.OD, fsu_pax.pax_nf)
            df2 = pd.DataFrame(list(zipped), 
             columns=['planned','avail','delta','arr_delay','dep_delay',
                      'sch_dep_tmz','od','pax']).sort_values(by='od')
            df2 = df2[df2['avail'] < delay]
            if df2.shape[0] == 0:
                df2_isempty = True
            else:
                df2_isempty = False

            if not printed_header and df2_isempty == False:
                st.write(df2)
                display(df2)
                printed_header = True

#---------------------------------------------------------
def stats(df):
    """ 
    Compute various statistics of FSU tables 

    Arguments: 
        df (DataFrame)

    Return: 
        no return
    """

    print("max fltnum: ", df.FLT_NUM.max())
    x = (df['SCH_DEP_DTZ'] == '2019/10/01').sum()
    print("nb flights on 2019/10/01: ", x)
    atb = (df['COUNT_ATB'] > 0).sum()
    gtb = (df['COUNT_GTB'] > 0).sum()
    print("ATB, GTB: ", atb)
    print("ATB, GTB: ", atb, gtb)
#---------------------------------------------------------
#---------------------------------------------------------
def handleCity(city, choice_ix, id_list, fsu, bookings_f, feeders, is_print=True, delay=45):
    # Need to increase efficiency

    # I need to return two structures: nodes and edges. 

    city_inbounds = findCity(bookings_f, city)
    #st.write("enter handleCity")

    if choice_ix == 'all':
        min_ix = 0
        max_ix = city_inbounds.shape[0]
    else:
        min_ix = choice_ix
        max_ix = choice_ix+1

    dfs = {}
    for which_ix in range(min_ix, max_ix):

        try:
            inbound = city_inbounds.iloc[which_ix].id_f
            fsu_inbound  = fsu.set_index('id').loc[inbound]
            #st.write("inbound= ", inbound)
            #st.write("city_inbounds= ", city_inbounds) # from Bookings_f database
        except:
            #st.write("first except")
            print("first except")
            continue

        try:
            fsu_inbound  = fsu.set_index('id').loc[inbound]
        except:
            # Sometimes the key "inbound" is not found
            print("HandleCity: second exception")
            continue

        inbound_arr_delay = fsu_inbound.ARR_DELAY_MINUTES
        
        # if the delay is negative, the plane arrived early, and the 
        # passengers have time to connect
        if inbound_arr_delay < 0:
            #continue # <<< ADD THIS BACK?
            pass

        outbounds = findOutboundIds(id_list, inbound).to_frame()
        outbounds['id_f'] = inbound
        #st.write("outbounds")
        #st.write(outbounds)

        #st.write("feeders: ", feeders.shape)  # <<<< THE DIFFERENCE
        #print("feeders: ", feeders.shape)

        # Create a unique id that combines inbound (feeder) and outbound flights
        # This will allow me to merge two files with feeder/non-feeder columns
        feeders_1 = feeders[feeders['id_f'] == inbound]
        feeders_1['id_f_nf'] = feeders_1['id_f'] + feeders_1['id_nf']

        # extract these outgoings from the FSU database
        # if outbounds has ids not in fsu, this approach will not work
        fsu_outbound = pd.merge(fsu, outbounds, how='inner', left_on='id', right_on='id_nf')
        fsu_outbound['id_f_nf'] = fsu_outbound['id_f'] + fsu_outbound['id_nf']

        #st.write("fsu_outbound: ", fsu_outbound.shape)
        #st.write("feeders_1: ", feeders_1.shape)
        fsu_outbound.to_csv("outbound_city.csv", index=0)
        feeders_1.to_csv("feeders_1_city.csv", index=0)

        fsu_pax = pd.merge(fsu_outbound, feeders_1, how='inner', left_on='id_f_nf', right_on='id_f_nf')
        #st.write("fsu_pax.shape: ", fsu_pax.shape)
        #st.write("fsu_outbound.head: ", fsu_outbound.head())
        #st.write("feeders_1.head: ", feeders_1.head())
        fsu_pax.drop_duplicates(inplace=True)
        #st.write(type(fsu_pax))

        # Compute connection time (inbound.IN - outbound.sch_dep)
        id_f = fsu_pax.id_f_x
        id_nf = fsu_pax.id_nf_x
        available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().IN_DTMZ) / 1e9 / 60
        #st.write("noID: available= ", available)
        planned   = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().SCH_ARR_DTMZ) / 1e9 / 60
        delta = planned - available   # = IN - SCH_ARR
        dep_delay = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_pax.IN_DTMZ  - fsu_pax.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        zipped = zip(id_nf, planned, available, delta, arr_delay, dep_delay, fsu_pax.SCH_DEP_TMZ, fsu_pax.OD, fsu_pax.pax_nf)

        #st.write("id_nf")
        #st.write(id_nf)
 
        columns=['id','planned','avail','delta','arr_delay','dep_delay','sch_dep_tmz','od','pax']
        df2 = pd.DataFrame(list(zipped), columns=columns).sort_values(by='od')
        df2 = df2[df2['avail'] < delay]

        if df2.shape[0] == 0:
            df2_isempty = True
        else:
            df2_isempty = False

        if df2_isempty == False:
            dfs[which_ix] = df2

        if is_print and df2_isempty == False:
            #st.write(df2)
            #display(df2)
            printed_header = True

    return dfs

#---------------------------------------------------------
def handleCityGraph(keep_early_arr, city, choice_ix, id_list, fsu, bookings_f, feeders, is_print=True, delay=45):
    # Need to increase efficiency

    #st.write("????????????????")
    #st.write("Enter handleCityGraph")
    #st.write("fsu.SCH_DEP_DTMZ", fsu.SCH_DEP_DTMZ)

    # I need to return two structures: nodes and edges. 
    # This pair of structures should be returned for each flight I am working with. 
    # Nodes are single IDs with arr and dep times, arr and dep delays. 
    # Edges are double IDs with PAX, rotation, and connection times. 

    #st.write("Enter handleCityGraph")

    # Find all inbound flights from a given city (other than PTY)
    city_inbounds = findCity(bookings_f, city)

    if choice_ix == 'all':
        min_ix = 0
        max_ix = city_inbounds.shape[0]
    else:
        min_ix = choice_ix
        max_ix = choice_ix+1

    # For each inbound flight, compute the corresponding outbound flights
    dfs = {}
    for which_ix in range(min_ix, max_ix):
        nodes = []

        try:
            inbound = city_inbounds.iloc[which_ix].id_f
            nodes.append(inbound)
            # keep: id_f only ==> a node
            fsu_inbound = fsu.set_index('id').loc[inbound]
        except:
            continue

        inbound_arr_delay = fsu_inbound.ARR_DELAY_MINUTES
        
        # if the arrival delay of the inbound is negative, the plane arrived early, and the 
        # passengers have time to connect
        if keep_early_arr == False and inbound_arr_delay < 0:
            continue

        # just a collection of 'id_nf' ==> nodes
        outbounds = findOutboundIds(id_list, inbound).to_frame()
        outbounds['id_f'] = inbound
        nodes.extend(outbounds['id_nf'].tolist())  # This is the list of nodes

        edges = outbounds['id_nf'].to_frame('e2')  # e2 is id_nf
        edges['e1'] = inbound   # e1 is id_f
        edges['id'] = edges['e1'] + '_' + edges['e2']

        # What is left to do is add the metadata to these lists
        # Nodes: the data comes from FSU files 
        # Edges: the data comes from PAX files

        # Create a unique id that combines inbound (feeder) and outbound flights
        # This will allow me to merge two files with feeder/non-feeder columns
        feeders_1 = feeders[feeders['id_f'] == inbound]
        feeders_1['id_f_nf'] = feeders_1['id_f'] + '_' + feeders_1['id_nf']

        # extract these outgoings from the FSU database
        # if outbounds has ids not in fsu, this approach will not work
        fsu_outbound = pd.merge(fsu, outbounds, how='inner', left_on='id', right_on='id_nf')
        fsu_outbound['id_f_nf'] = fsu_outbound['id_f'] + '_' + fsu_outbound['id_nf']

        fsu_pax = pd.merge(fsu_outbound, feeders_1, how='inner', left_on='id_f_nf', right_on='id_f_nf')
        #fsu_outbound.to_csv("outbound_cityGraph.csv", index=0)
        fsu_pax.drop_duplicates(inplace=True)

        # Compute connection time (inbound.IN - outbound.sch_dep)
        id_f = fsu_pax.id_f_x
        id_nf = fsu_pax.id_nf_x
        # Node metadata
        dep_delay = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_pax.IN_DTMZ  - fsu_pax.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        od = fsu_pax.OD
        node_nf_dict = {'id':id_nf, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od}
        d_nf = pd.DataFrame(node_nf_dict)

        # Add feeder row
        # Find inbound in FSU data
        # fsu_inbound is a Series. Another way to access : fsu_inbound.loc['SCH_DEP_DTMZ',0]
        dep_delay = (fsu_inbound.OUT_DTMZ - fsu_inbound.transpose().SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_inbound.IN_DTMZ  - fsu_inbound.transpose().SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        od = fsu_inbound.OD
        row_f = {'id':inbound, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od}
        d_nf.loc[-1] = row_f
        # drop=True: do not keep the new index column created by default
        node_df = d_nf.sort_index().reset_index(drop=True)

        # The first node is the feeder
        # All the other nodes are the outbounds

        # Create Graph edges and metadata
        id_f_nf = id_f + "_" + id_nf
        #st.write("handleGraph, fsu_inbound= ", fsu_inbound)
        #st.write("IN_DTMZ: ", fsu_inbound.IN_DTMZ)
        #st.write("SCH_DEP_DTMZ: ", fsu_outbound.SCH_DEP_DTMZ)
        # Not clear why transpose is no longer needed. 
        available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.IN_DTMZ) / 1e9 / 60
        planned = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.SCH_ARR_DTMZ) / 1e9 / 60
        #st.write("available= ", available)
        #available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().IN_DTMZ) / 1e9 / 60
        #planned = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.transpose().SCH_ARR_DTMZ) / 1e9 / 60
        pax_id_nf = fsu_pax.id_nf_y
        pax_id_f  = fsu_pax.id_f_y
        pax_avail = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.transpose().IN_DTMZ) / 1e9 / 60
        pax_planned = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.transpose().SCH_ARR_DTMZ) / 1e9 / 60
        dfx = pd.DataFrame([pax_id_f, pax_id_nf, pax_avail, pax_planned]).transpose()

        fsux = fsu[fsu['id'] == '2019/10/01SJOPTY10:29459']
        fsuy = fsu[fsu['id'] == '2019/10/01PTYTPA14:12393']

        delta = planned - available   # = IN - SCH_ARR
        edge_nf_zip = zip(available, planned, delta)
        id_f = fsu_pax['id_f_y']
        id_nf = fsu_pax['id_nf_y']
        id_f_nf = fsu_pax['id_f_nf']
        #st.write("No ID, fsu_pax: ", fsu_pax.shape)
        edge_df = pd.DataFrame()
        edge_df = pd.concat([edge_df, id_f_nf, id_f, id_nf], axis=1)
        #st.write("No ID, edge_df: ", edge_df.shape)

        ## Reorder the columns for clarity
        edge_df['avail'] = available
        edge_df['planned'] = planned
        edge_df['delta'] = delta
        edge_df['pax'] = fsu_pax.pax_nf

        ## EDGE correct. Now add metadata: avail, planned, delta
 
        # Remove edges and nodes for flights with less available connection time than `delay`
        # (I could either simplify the graph, or use brushing in the graph. Or both.)
        # Let us do both. Simplification in this method, and brushing in Altair. 

        #st.write(node_df.columns)
        #st.write(edge_df.columns)
        # 1. find all nodes to remove
        # The passengers that "could" miss their flights have less available time than needed. 
        # We keep the flights that potentially have the most impact on the network 
        ids_nf_to_keep = edge_df[edge_df['avail'] < delay]['id_nf_y']
        #st.write("No ID, edge_df after filtering delays: ", edge_df.shape)
        #st.write("ID, ids_nf_to_keep: ", ids_nf_to_keep)
        #st.write("node_df: ", node_df)
        #st.write("edge_df= ", edge_df)
        #st.write("delay= ", delay)
        #st.write("handleCitiesGraph, ids_nf_to_keep: ", ids_nf_to_keep)  # # EMPTY. WHY? 

        #st.write("ids_nf_to_keep: ", ids_nf_to_keep)

        # 2. delete nodes from node DataFrame
        node_df = node_df.set_index('id').loc[ids_nf_to_keep,:].reset_index()
        # Add back the first row that is the feeder (it stays)
        #st.write("NoID, row_f= ", row_f)
        node_df.loc[-1] = row_f
        node_df = node_df.sort_index().reset_index(drop=True)

        # 3. delete edges from edge DataFrame
        edge_df = edge_df.set_index('id_nf_y').loc[ids_nf_to_keep,:].reset_index()

        #st.write("NoID, node_df: ", node_df)
        #st.write("NoID, edge_df: ", edge_df)
        #st.write(node_df.shape, edge_df.shape)


        # Only the first ix
        return node_df, edge_df

        continue

        columns=['id','planned','avail','delta','arr_delay','dep_delay','sch_dep_tmz','od','pax']
        df2 = pd.DataFrame(list(zipped), columns=columns).sort_values(by='od')
        df2 = df2[df2['avail'] < delay]

        if df2.shape[0] == 0:
            df2_isempty = True
        else:
            df2_isempty = False

        if df2_isempty == False:
            dfs[which_ix] = df2

        if is_print and df2_isempty == False:
            #st.write(df2)
            #display(df2)
            printed_header = True

    return dfs

#---------------------------------------------------------
def handleCityGraphId(flight_id, keep_early_arr, only_keep_late_dep, id_list, fsu, bookings_f, feeders, is_print=True, delay=45, flight_id_level=0, debug=False):
    """
    Given an inbound flight to PTY return the corresponding outbound flighs
    Return a tuple of Dataframes with node and edges

    Arguments

      flight_id_level: level of flight_id in the graph network. The root has level zero. Children of flight_id have level 1, grandchildren of flight_id have level 2. Each leg of a flight increases the level by 1. 
    """
    #st.write("enter handleCityGraphId")

    # I need to return two structures: nodes and edges. 
    # This pair of structures should be returned for each flight I am working with. 
    # Nodes are single IDs with arr and dep times, arr and dep delays. 
    # Edges are double IDs with PAX, rotation, and connection times. 

    #st.write("Enter handleCityGraph")

    inbound = flight_id
    node_df = pd.DataFrame()
    edge_df = pd.DataFrame()
    #city_inbounds = findCity(bookings_f, city)

    """
    if choice_ix == 'all':
        min_ix = 0
        max_ix = city_inbounds.shape[0]
    else:
        min_ix = choice_ix
        max_ix = choice_ix+1
    """
    min_ix, max_ix = 0, 1  # single element in the loop, so single flight_id

    # For each inbound flight, compute the corresponding outbound flights
    for which_ix in range(min_ix, max_ix):
        nodes = []

        inbound = pd.DataFrame({'id':[inbound]})  # New, created from method argument
        try:
            nodes.append(inbound)
            fsu_inbound = fsu[fsu['id'] == inbound['id'].values[0]]
        except:
            #st.write("except")
            continue

        inbound_arr_delay = fsu_inbound.ARR_DELAY_MINUTES.values[0]
        
        # if the arrival delay of the inbound is negative, the plane arrived early, and the 
        # passengers have time to connect
        #st.write("inbound_arr_delay= ", inbound_arr_delay)  # DF

        # Do not keep early arrivals
        # If the inbound (into PTY) is early, ignore subsequent flights
        if keep_early_arr == False and inbound_arr_delay < 0:
            #st.write("continue")
            ## Must keep keep_early_arrivals to TRUE for now. 2021-06-07. 
            continue


        # just a collection of 'id_nf' ==> nodes
        inbound = inbound['id'].values[0]  # Series convert to list using .values
        outbounds = findOutboundIds(id_list, inbound).to_frame()
        outbounds['id_f'] = inbound
        nodes.extend(outbounds['id_nf'].tolist())  # This is the list of nodes

        edges = outbounds['id_nf'].to_frame('e2')  # e2 is id_nf
        edges['e1'] = inbound   # e1 is id_f
        edges['id'] = edges['e1'] + '_' + edges['e2']

        # What is left to do is add the metadata to these lists
        # Nodes: the data comes from FSU files 
        # Edges: the data comes from PAX files

        # Create a unique id that combines inbound (feeder) and outbound flights
        # This will allow me to merge two files with feeder/non-feeder columns
        feeders_1 = feeders[feeders['id_f'] == inbound]
        #st.write("handle*Id, feeders_1= ", feeders_1)

        feeders_1['id_f_nf'] = feeders_1['id_f'] + '_' + feeders_1['id_nf']

        # extract these outgoings from the FSU database
        # if outbounds has ids not in fsu, this approach will not work
        fsu_outbound = pd.merge(fsu, outbounds, how='inner', left_on='id', right_on='id_nf')
        fsu_outbound['id_f_nf'] = fsu_outbound['id_f'] + '_' + fsu_outbound['id_nf']
        #st.write("fsu_outbound= ", fsu_outbound)

        fsu_pax = pd.merge(fsu_outbound, feeders_1, how='inner', left_on='id_f_nf', right_on='id_f_nf')
        #st.write("top level: fsu_pax.shape: ", fsu_pax.shape)
        #fsu_outbound.to_csv("outbound_cityGraph.csv", index=0)
        fsu_pax.drop_duplicates(inplace=True)
        #st.write("fsu_pax= ", fsu_pax)

        # Compute connection time (inbound.IN - outbound.sch_dep)
        id_f = fsu_pax.id_f_x
        id_nf = fsu_pax.id_nf_x
        # Node metadata
        fsu_pax['dep_delay'] = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60

        if only_keep_late_dep: 
            fsu_pax = fsu_pax[fsu_pax['dep_delay'] > 0]

        dep_delay = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_pax.IN_DTMZ  - fsu_pax.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        flt_num = fsu_pax.FLT_NUM
        tail = fsu_pax.TAIL
        od = fsu_pax.OD
        sch_dep_tmz = fsu_pax.SCH_DEP_TMZ
        sch_arr_tmz = fsu_pax.SCH_ARR_TMZ
        node_nf_dict = {'id':id_nf, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od, 'FLT_NUM':flt_num, 'TAIL':tail, 'SCH_DEP_TMZ':sch_dep_tmz, 'SCH_ARR_TMZ':sch_arr_tmz}
        d_nf = pd.DataFrame(node_nf_dict)
        #st.write("d_nf= ", d_nf)

        # Add feeder row
        # Find inbound in FSU data
        # fsu_inbound is a Series. Another way to access : fsu_inbound.loc['SCH_DEP_DTMZ',0]
        #st.write("fsu_inbound= ", fsu_inbound)
        # I removed the transpose. Not clear why. 
        #dep_delay = (fsu_inbound.OUT_DTMZ - fsu_inbound.transpose().SCH_DEP_DTMZ) / 1e9 / 60
        dep_delay = (fsu_inbound.OUT_DTMZ - fsu_inbound.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_inbound.IN_DTMZ  - fsu_inbound.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        dep_delay = dep_delay.values[0]  # Must fix this. Not clear why needed here.
        arr_delay = arr_delay.values[0]

        od = fsu_inbound.OD.values[0]  # Series
        flt_num = fsu_inbound.FLT_NUM.values[0]
        tail = fsu_inbound.TAIL.values[0]
        
        #### IT is zero on second level of inbound flights to PTY. WHY? (2021-06-09)
        #st.write("fsu_pax.shape: ", fsu_pax.shape)
        if fsu_pax.shape[0] == 0:
            continue

        #st.write("fsu_inbound: ", fsu_inbound.SCH_DEP_TMZ)
        #st.write("inbound: " , flight_id)
        #st.write("fsu_pax: ", fsu_pax[['id','SCH_DEP_TMZ']])
        #sch_dep_tmz = fsu_pax.SCH_DEP_TMZ.values[0]
        #sch_arr_tmz = fsu_pax.SCH_ARR_TMZ.values[0]
        sch_dep_tmz = fsu_inbound.SCH_DEP_TMZ.values[0]
        sch_arr_tmz = fsu_inbound.SCH_ARR_TMZ.values[0]

        row_f = {'id':inbound, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od, 'lev':flight_id_level, 'FLT_NUM':flt_num, 'TAIL':tail, 'SCH_DEP_TMZ':sch_dep_tmz, 'SCH_ARR_TMZ':sch_arr_tmz}
        #st.write(row_f)
        #print(type(fsu_inbound))

        d_nf.loc[-1] = row_f
        # drop=True: do not keep the new index column created by default
        node_df = d_nf.sort_index().reset_index(drop=True)
        node_df.loc[:,'lev'] = flight_id_level
        node_df.loc[1:,'lev'] = flight_id_level + 1

        # The first node is the feeder
        # All the other nodes are the outbounds

        # Create Graph edges and metadata
        id_f_nf = id_f + "_" + id_nf
        # Why isn't IN_DTMZ a scalar like in the method handleCitiesGraph()?
        available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.IN_DTMZ.values[0]) / 1e9 / 60
        planned = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.SCH_ARR_DTMZ.values[0]) / 1e9 / 60
        #if debug:
            #st.write("id: ", fsu_inbound.id)
            #st.write("inbound arrival: ", fsu_inbound.SCH_ARR_TMZ)
            #st.write("outbound departure: ", fsu_outbound[['id','SCH_DEP_TMZ']])
            #st.write("planned: ", planned);

        #pax_id_nf = fsu_pax.id_nf_y
        #pax_id_f  = fsu_pax.id_f_y
        #pax_avail = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.IN_DTMZ.values[0]) / 1e9 / 60
        #pax_planned = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.SCH_ARR_DTMZ) / 1e9 / 60
        #if debug:
            #st.write("pax planned: ", pax_planned);

        #st.write("planned, pax_planned: ", planned, pax_planned)
        #st.write("available, pax_avail: ", available, pax_avail)
        #dfx = pd.DataFrame([pax_id_f, pax_id_nf, pax_avail, pax_planned]).transpose()
        #st.write("1 node_df: ", node_df)

        fsux = fsu[fsu['id'] == '2019/10/01SJOPTY10:29459']
        fsuy = fsu[fsu['id'] == '2019/10/01PTYTPA14:12393']

        delta = planned - available   # = IN - SCH_ARR
        edge_nf_zip = zip(available, planned, delta)
        id_f = fsu_pax['id_f_y']
        id_nf = fsu_pax['id_nf_y']
        id_f_nf = fsu_pax['id_f_nf']
        edge_df = pd.DataFrame()
        edge_df = pd.concat([edge_df, id_f_nf, id_f, id_nf], axis=1)
        #st.write("ID, edge_df: ", edge_df.shape)
        ## Reorder the columns for clarity
        edge_df['avail'] = available
        edge_df['planned'] = planned
        edge_df['delta'] = delta
        edge_df['pax'] = fsu_pax.pax_nf
        #st.write("ID, fsu_pax: ", fsu_pax.shape)
        ## EDGE correct. Now add metadata: avail, planned, delta
 
        # Remove edges and nodes for flights with less available connection time than `delay`
        # (I could either simplify the graph, or use brushing in the graph. Or both.)
        # Let us do both. Simplification in this method, and brushing in Altair. 

        #st.write(node_df.columns)
        #st.write(edge_df.columns)
        # 1. find all nodes to remove
        # The passengers that "could" miss their flights have less available time than needed. 
        # We keep the flights that potentially have the most impact on the network 
        #st.write("delay= ", delay)

        ids_nf_to_keep = edge_df[edge_df['avail'] < delay]['id_nf_y']

        #st.write("ID, ids_nf_to_keep: ", ids_nf_to_keep)
        #st.write("ID, edge_df after filtering delays: ", edge_df.shape)

        #st.write("ids_nf_to_keep: ", ids_nf_to_keep)

        # 2. delete nodes from node DataFrame
        #st.write("node_df: ", node_df)
        #st.write("edge_df= ", edge_df)
        #st.write("delay= ", delay)
        node_df = node_df.set_index('id').loc[ids_nf_to_keep,:].reset_index()

        #st.write("3 node_df: ", node_df)  # EMPTY
        # Add back the first row that is the feeder (it stays)
        node_df.loc[-1] = row_f
        node_df = node_df.sort_index().reset_index(drop=True)


        # 3. delete edges from edge DataFrame
        edge_df = edge_df.set_index('id_nf_y').loc[ids_nf_to_keep,:].reset_index()

        #st.write("ID, last node_df: ", node_df)
        #st.write("ID, last edge_df: ", edge_df)

        #st.write("node_df: ", node_df)
        #st.write("edge_df: ", edge_df)
        #if nb_search > 0: st.stop()  # DEBUGGING
        #st.write(node_df.shape, edge_df.shape)

        # Only the first ix
        #st.write(node_df)
        #st.write(edge_df)

        #if debug: 
            #st.write("edge_df: ", edge_df)

        #st.write("handle: node_df: ", node_df)
    return node_df, edge_df

#---------------------------------------------------------
def handleCityGraphIdLev2_xxx(flight_id, keep_early_arr, id_list, fsu, bookings_f, feeders, is_print=True, delay=45, flight_id_level=0):
    """
    Given an inbound flight to PTY return the corresponding outbound flighs
    Return a tuple of Dataframes with node and edges

    >>> Second tier flights <<<

    Arguments

      flight_id_level: level of flight_id in the graph network. The root has level zero. Children of flight_id have level 1, grandchildren of flight_id have level 2. Each leg of a flight increases the level by 1. 
    """
    #st.write("enter handleCityGraphId")

    # I need to return two structures: nodes and edges. 
    # This pair of structures should be returned for each flight I am working with. 
    # Nodes are single IDs with arr and dep times, arr and dep delays. 
    # Edges are double IDs with PAX, rotation, and connection times. 

    #st.write("Enter handleCityGraph")

    inbound = flight_id
    #city_inbounds = findCity(bookings_f, city)

    """
    if choice_ix == 'all':
        min_ix = 0
        max_ix = city_inbounds.shape[0]
    else:
        min_ix = choice_ix
        max_ix = choice_ix+1
    """
    min_ix, max_ix = 0, 1

    # For each inbound flight, compute the corresponding outbound flights
    for which_ix in range(min_ix, max_ix):
        nodes = []

        inbound = pd.DataFrame({'id':[inbound]})  # New, created from method argument

        try:
            nodes.append(inbound)
            fsu_inbound = fsu[fsu['id'] == inbound['id'].values[0]]
        except:
            print("except")
            continue

        inbound_arr_delay = fsu_inbound.ARR_DELAY_MINUTES.values[0]
        
        # if the arrival delay of the inbound is negative, the plane arrived early, and the 
        # passengers have time to connect

        if keep_early_arr == False and inbound_arr_delay < 0:
            #st.write("continue")
            ## Must keep keep_early_arrivals to TRUE for now. 2021-06-07. 
            continue

        # just a collection of 'id_nf' ==> nodes
        inbound = inbound['id'].values[0]  # Series convert to list using .values
        outbounds = findOutboundIds(id_list, inbound).to_frame()
        outbounds['id_f'] = inbound
        nodes.extend(outbounds['id_nf'].tolist())  # This is the list of nodes

        edges = outbounds['id_nf'].to_frame('e2')  # e2 is id_nf
        edges['e1'] = inbound   # e1 is id_f
        edges['id'] = edges['e1'] + '_' + edges['e2']


        # What is left to do is add the metadata to these lists
        # Nodes: the data comes from FSU files 
        # Edges: the data comes from PAX files

        # Create a unique id that combines inbound (feeder) and outbound flights
        # This will allow me to merge two files with feeder/non-feeder columns
        feeders_1 = feeders[feeders['id_f'] == inbound]
        feeders_1['id_f_nf'] = feeders_1['id_f'] + '_' + feeders_1['id_nf']

        # extract these outgoings from the FSU database
        # if outbounds has ids not in fsu, this approach will not work
        fsu_outbound = pd.merge(fsu, outbounds, how='inner', left_on='id', right_on='id_nf')
        fsu_outbound['id_f_nf'] = fsu_outbound['id_f'] + '_' + fsu_outbound['id_nf']

        fsu_pax = pd.merge(fsu_outbound, feeders_1, how='inner', left_on='id_f_nf', right_on='id_f_nf')
        #st.write("top level: fsu_pax.shape: ", fsu_pax.shape)
        fsu_pax.drop_duplicates(inplace=True)

        # Compute connection time (inbound.IN - outbound.sch_dep)
        id_f = fsu_pax.id_f_x
        id_nf = fsu_pax.id_nf_x
        # Node metadata
        dep_delay = (fsu_pax.OUT_DTMZ - fsu_pax.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_pax.IN_DTMZ  - fsu_pax.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        flt_num = fsu_pax.FLT_NUM
        tail = fsu_pax.TAIL
        od = fsu_pax.OD
        sch_dep_tmz = fsu_pax.SCH_DEP_TMZ
        sch_arr_tmz = fsu_pax.SCH_ARR_TMZ
        node_nf_dict = {'id':id_nf, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od, 'FLT_NUM':flt_num, 'TAIL':tail, 'SCH_DEP_TMZ':sch_dep_tmz, 'SCH_ARR_TMZ':sch_arr_tmz}
        d_nf = pd.DataFrame(node_nf_dict)
        #st.write("d_nf= ", d_nf)

        # Add feeder row
        # Find inbound in FSU data
        # fsu_inbound is a Series. Another way to access : fsu_inbound.loc['SCH_DEP_DTMZ',0]
        #st.write("fsu_inbound= ", fsu_inbound)
        # I removed the transpose. Not clear why. 
        #dep_delay = (fsu_inbound.OUT_DTMZ - fsu_inbound.transpose().SCH_DEP_DTMZ) / 1e9 / 60
        dep_delay = (fsu_inbound.OUT_DTMZ - fsu_inbound.SCH_DEP_DTMZ) / 1e9 / 60
        arr_delay = (fsu_inbound.IN_DTMZ  - fsu_inbound.SCH_ARR_DTMZ) / 1e9 / 60   # outbound
        dep_delay = dep_delay.values[0]  # Must fix this. Not clear why needed here.
        arr_delay = arr_delay.values[0]

        od = fsu_inbound.OD.values[0]  # Series
        flt_num = fsu_inbound.FLT_NUM.values[0]
        tail = fsu_inbound.TAIL.values[0]
        
        #### IT is zero on second level of inbound flights to PTY. WHY? (2021-06-09)
        #st.write("fsu_pax.shape: ", fsu_pax.shape)
        if fsu_pax.shape[0] == 0:
            continue

        sch_dep_tmz = fsu_pax.SCH_DEP_TMZ.values[0]
        sch_arr_tmz = fsu_pax.SCH_ARR_TMZ.values[0]

        row_f = {'id':inbound, 'arr_delay':arr_delay, 'dep_delay':dep_delay, 'od':od, 'lev':flight_id_level, 'FLT_NUM':flt_num, 'TAIL':tail, 'SCH_DEP_TMZ':sch_dep_tmz, 'SCH_ARR_TMZ':sch_arr_tmz}
        #st.write(row_f)

        d_nf.loc[-1] = row_f
        # drop=True: do not keep the new index column created by default
        node_df = d_nf.sort_index().reset_index(drop=True)
        node_df.loc[:,'lev'] = flight_id_level
        node_df.loc[1:,'lev'] = flight_id_level + 1

        # The first node is the feeder
        # All the other nodes are the outbounds

        # Create Graph edges and metadata
        id_f_nf = id_f + "_" + id_nf
        # Why isn't IN_DTMZ a scalar like in the method handleCitiesGraph()?
        available = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.IN_DTMZ.values[0]) / 1e9 / 60
        planned = (fsu_outbound.SCH_DEP_DTMZ - fsu_inbound.SCH_ARR_DTMZ.values[0]) / 1e9 / 60

        pax_id_nf = fsu_pax.id_nf_y
        pax_id_f  = fsu_pax.id_f_y
        pax_avail = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.IN_DTMZ.values[0]) / 1e9 / 60
        pax_planned = (fsu_pax.SCH_DEP_DTMZ - fsu_inbound.SCH_ARR_DTMZ) / 1e9 / 60

        #st.write("planned, pax_planned: ", planned, pax_planned)
        #st.write("available, pax_avail: ", available, pax_avail)
        dfx = pd.DataFrame([pax_id_f, pax_id_nf, pax_avail, pax_planned]).transpose()
        #st.write("1 node_df: ", node_df)

        fsux = fsu[fsu['id'] == '2019/10/01SJOPTY10:29459']
        fsuy = fsu[fsu['id'] == '2019/10/01PTYTPA14:12393']

        delta = planned - available   # = IN - SCH_ARR
        edge_nf_zip = zip(available, planned, delta)
        id_f = fsu_pax['id_f_y']
        id_nf = fsu_pax['id_nf_y']
        id_f_nf = fsu_pax['id_f_nf']
        edge_df = pd.DataFrame()
        edge_df = pd.concat([edge_df, id_f_nf, id_f, id_nf], axis=1)
        #st.write("ID, edge_df: ", edge_df.shape)
        ## Reorder the columns for clarity
        edge_df['avail'] = available
        edge_df['planned'] = planned
        edge_df['delta'] = delta
        edge_df['pax'] = fsu_pax.pax_nf
        #st.write("ID, fsu_pax: ", fsu_pax.shape)
        ## EDGE correct. Now add metadata: avail, planned, delta
 
        # Remove edges and nodes for flights with less available connection time than `delay`
        # (I could either simplify the graph, or use brushing in the graph. Or both.)
        # Let us do both. Simplification in this method, and brushing in Altair. 

        #st.write(node_df.columns)
        #st.write(edge_df.columns)
        # 1. find all nodes to remove
        # The passengers that "could" miss their flights have less available time than needed. 
        # We keep the flights that potentially have the most impact on the network 
        #st.write("delay= ", delay)

        ids_nf_to_keep = edge_df[edge_df['avail'] < delay]['id_nf_y']

        #st.write("ID, ids_nf_to_keep: ", ids_nf_to_keep)
        #st.write("ID, edge_df after filtering delays: ", edge_df.shape)

        #st.write("ids_nf_to_keep: ", ids_nf_to_keep)

        # 2. delete nodes from node DataFrame
        #st.write("node_df: ", node_df)
        #st.write("edge_df= ", edge_df)
        #st.write("delay= ", delay)
        node_df = node_df.set_index('id').loc[ids_nf_to_keep,:].reset_index()

        #st.write("3 node_df: ", node_df)  # EMPTY
        # Add back the first row that is the feeder (it stays)
        node_df.loc[-1] = row_f
        node_df = node_df.sort_index().reset_index(drop=True)


        # 3. delete edges from edge DataFrame
        edge_df = edge_df.set_index('id_nf_y').loc[ids_nf_to_keep,:].reset_index()

        #st.write("ID, last node_df: ", node_df)
        #st.write("ID, last edge_df: ", edge_df)

        #st.write("node_df: ", node_df)
        #st.write("edge_df: ", edge_df)
        #st.write(node_df.shape, edge_df.shape)

        # Only the first ix
        #st.write(node_df)
        #st.write(edge_df)
        return node_df, edge_df

#---------------------------------------------------------
def getReturnFlights(pairs, keep_early_arr, keep_late_dep, node_df, edge_df, dct, fsu, flight_id_level=0):
    # Scan each outbound flight on the graph and find the corresponding in inbound flight
    # pairs: pairs of ids (flight_from, flight_to) with identical tails
    #st.write("enter getReturn")
    #st.write("*** getReturnFlights ***")
    outbound_ids = node_df['id'].to_list()[1:]
    #st.write("outbound_ids: ", outbound_ids)
    #st.write("0 node_df: ", node_df)
    #st.write("0 edge_df: ", edge_df)
    # Rewrite with map or apply. Do not know how as yet. 
    fsu1 = fsu.sort_values('id').set_index('id', drop=False)


    # outbound from PTY
    for i, outbound in enumerate(outbound_ids): 
        #ids = list(fsu1.index)

        #st.write("corresponding connection (same tail): ")
        city = outbound[13:16]

        try:
            next_fid  = pairs.loc[outbound]
            #st.write("outbound, next_fid: ", outbound[10:16], next_fid.id2[10:16])
        except:
            continue

        #inbounds_df = dct[outbound]

        #if inbounds_df.shape[0] > 0:
        if True:
            # First departure from city X
            #inbound = inbounds_df.iloc[0].id
            inbound = next_fid.id2
            #st.write("-- inbound= ", inbound)

            # Perhaps add a channel for the line type (dashed or solid)
            # id_f refers to the source of the edge
            # id_nf refers to the target
            # id_f_nf has no meaning in this context
            rec = fsu1.loc[inbound]
            node_df.loc[10000+i,'id'] = inbound
            node_df.loc[10000+i,'lev'] = flight_id_level
            node_df.loc[10000+i,'od'] = inbound[10:16]
            node_df.loc[10000+i,'FLT_NUM'] = fsu1.loc[inbound].FLT_NUM
            node_df.loc[10000+i,'TAIL'] = fsu1.loc[inbound].TAIL
            node_df.loc[10000+i,'SCH_DEP_TMZ'] = rec.SCH_DEP_TMZ
            node_df.loc[10000+i,'SCH_ARR_TMZ'] = rec.SCH_ARR_TMZ
            node_df.loc[10000+i,'dep_delay'] =  \
                (rec.OUT_DTMZ - rec.SCH_DEP_DTMZ) / 1e9 / 60
            node_df.loc[10000+i,'arr_delay'] = \
                (rec.IN_DTMZ  - rec.SCH_ARR_DTMZ) / 1e9 / 60 
            #st.write("inbound[10:16]= ", inbound[10:16])
            #st.write("xxx: ", node_df)

            #st.write(outbound)
            #st.write(inbound)

            # Connection in city X
            rec_in  = fsu1.loc[outbound] # inbound to city X
            rec_out = fsu1.loc[inbound] # outbound from city X
            available = (rec_out.SCH_DEP_DTMZ - rec_in.IN_DTMZ) / 1e9 / 60
            planned   = (rec_out.SCH_DEP_DTMZ - rec_in.SCH_ARR_DTMZ) / 1e9 / 60
            #if outbound == '2019/10/01PTYHAV12:32371':
                #st.write("...inbound: ", inbound)
                #st.write("...rec_out: ", rec_out)
                #st.write("...rec_in: ", rec_in)
                #st.write("...rec_out.SCH_DEP_TMZ: ", rec_out.SCH_DEP_TMZ)
                #st.write("...rec_in.SCH_ARR_TMZ: ", rec_in.SCH_ARR_TMZ)
            delta = planned - available

            edge_df.loc[10000+i,['id_f_y','id_nf_y','id_f_nf']] = [outbound, inbound, np.nan]
            edge_df.loc[10000+i,['planned','delta']] = [planned, delta]
            edge_df.loc[10000+i,'avail'] = available  # does not work in previous line. WHY NOT? 
            edge_df.loc[10000+i,'pax'] = -1  


    node_df = node_df.reset_index(drop=True)
    edge_df = edge_df.reset_index(drop=True)

    #st.write("return from getReturnFlight, node_df: ", node_df)
    #st.write("return from getReturnFlight, edge_df: ", edge_df)
    return node_df, edge_df

#---------------------------------------------------------
def timestampToDateTimeUTC(ts):
    # ts: timestamp in seconds
    #dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
    return dt_tm[0:10], dt_tm[11:16]
#---------------------------------------------------------
def timestampToDateTimePTY(ts):
    # ts: timestamp in seconds
    dt_tm = timestampToDateTimeUTC(ts + 5*3600)
    return dt_tm
#---------------------------------------------------------
def dateTimeUTCToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0000"

    # convert string to datetimeformat
    date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#---------------------------------------------------------
def dateTimePTYToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0500"  # PTY is 5 hrs ahead of UTC

    # convert string to datetimeformat
    date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#---------------------------------------------------------
def filterDateRange(dateRange, df, attr='SCH_DEP_DTMZ', sort_attr='SCH_DEP_DTZ'):
    """
    Given a range of dates and time at PTY, return the frames of df with 
    departure time in that range. 

    Arguments:
    dateRange: [date1, time1, date2, time2], where
        date1: lower date in the format 'yyyy/mm/dd'
        date2: upper date in the format 'yyyy/mm/dd'
        time1: lower time in the format 'hh:mm'
        time2: upper time in the format 'hh:mm'
    df: DataFrame with the field: SCH_DEP_DTMZ (in nanoseconds)
    attr ['SCH_DEP_DTMZ']: the column label (timestamp) used for the filter action
    sort_attr ['SCH_DEP_DTZ']: the column label to sort on

    The input arguments are not modified by the function

    Return: the subset of the dataframe that satisfies the constraints 
        (datetime >= (date1,time1)
         datetime <= (date2,time2)
   """
    date1, time1, date2, time2 = dateRange
    ts1 = dateTimePTYToTimestamp(date1, time1) * 1e9
    ts2 = dateTimePTYToTimestamp(date2, time2) * 1e9
    filtered_df = df[(df[attr] >= ts1) & (df[attr] <= ts2)].sort_values(sort_attr)
    return filtered_df
#------------------------------------------------------------
