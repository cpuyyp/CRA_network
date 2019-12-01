
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from function_library import *

def createSenderMatrix(l_sent, l_from, unique_senders, name2id, save=True):
    df_Sent = pd.DataFrame(data=l_sent,columns = ['Sent'])
    sender_by_year = np.zeros((len(unique_senders),6))
    for year in range(2012,2018):
        index1 = df_Sent> datetime( year, 1, 1, 0, 0, 0)
        index2 = df_Sent< datetime( year+1, 1, 1, 0, 0, 0)
        index = (index1&index2).values.flatten().tolist()
        l_from_sliced = np.array(l_from)[index].tolist()
        # cannot directly use restrictEmailsToYears function because it's not returning the index
        col = year-2012
        for sender in l_from_sliced:
            row = name2id[sender[2]]
            sender_by_year[row,col] += 1

    sender_by_year_temp = sender_by_year.T.tolist()
    sender_by_year_temp.insert(0,unique_senders)
    sender_by_year_temp = list(map(list, zip(*sender_by_year_temp)))
    df_sender_by_year = pd.DataFrame(data = sender_by_year_temp, columns=['Senders',2012,2013,2014,2015,2016,2017])
    if save == True:
        df_sender_by_year.to_csv('sender_by_year.csv')
    return df_sender_by_year
