
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from function_library import *

#----------------------------------------------------------------------
def createPeopleMatrix(l_people, l_time, start, end, save_to_file=None):
    # l_people can be l_to, l_cc, l_sent
    # l_time can be l_year, l_month, l_week, l_weekday, l_hour
    # start and end should be in appropriate range,
    # e.g. if use l_year, start and end are in range [2012,2018]
    # save_to_file is the name of csv that you want, if it's None, not saving

    # if the list of people is not a unique person, e.g. l_to and l_cc
    if type(l_people[0]) == list:
        unique_people = set()
        for i in range(len(l_people)):
            for lst in l_people[i]:
                unique_people.add(lst)
        unique_people = list(unique_people)
        unique_people.sort()
    # otherwise, the list of people must be l_from
    else:
        unique_people = set()
        for i in range(len(l_people)):
            unique_people.add(l_people[i])
        unique_people = list(unique_people)
        unique_people.sort()

    # initialize matrix and variables
    name2id, id2name = nameToIndexDict(unique_people)
    num_timeslice = end - start
    people_by_time = np.zeros((len(unique_people), num_timeslice))
    # use dataframe because it's easier for get index
    df_time = pd.DataFrame(data=l_time,columns = ['Sent'])

    # iterate through all the columns
    for time_point in range(start,end):
        # pick out rows that satisfy the condition
        index = (df_time == time_point).values.flatten()
        l_people_sliced = np.array(l_people)[index].tolist()
        l_people_sliced = standardize_triplet(l_people_sliced)
        col = time_point-start
        for people in l_people_sliced:
            # if the list of people is l_from
            if type(people[0]) == str:
                row = name2id[people]
                people_by_time[row,col] += 1
            # otherwise, it is l_to or l_cc
            else:
                for person in people:
                    row = name2id[person]
                    people_by_time[row,col] += 1


    # insert people's names to the first column
    people_by_time_temp = people_by_time.T.tolist()
    people_by_time_temp.insert(0,unique_people)
    people_by_time_temp = list(map(list, zip(*people_by_time_temp)))
    columns = ['people'] + np.arange(start,end).tolist()
    df_people_by_time= pd.DataFrame(data = people_by_time_temp, columns=columns)
    if save_to_file != None:
        df_people_by_time.to_csv(save_to_file+'.csv')
    return df_people_by_time


#----------------------------------------------------------------------
def plot_barchart_by_time(df_people_by_time, time, top = 20, sortby = 'total', show_label = 'first', remove_blank = True, save_to_file=None):
    # df_people_by_time is got from the createPeopleMatrix function
    # time should be a valid timepoint in the right range
    # top shows top n people
    # the order is controled by sortby.
    # If sortby == 'total', then it is using the total number of emails in all time,
    # else use the selected column
    # show_label can show first name or last name or entire name or email or triplet
    # remove_blank will remove the empty triplet ('', '', '')

    if remove_blank == True:
        index = df_people_by_time.index[df_people_by_time['people'] == ('', '', '')]
        df_people_by_time = df_people_by_time.drop(index)
        df_people_by_time = df_people_by_time.reset_index(drop=True)
    if sortby == 'total':
        df_people_by_time['sum'] = df_people_by_time.sum(axis=1)
        df_people_by_time = df_people_by_time.sort_values(ascending=False, by='sum')
        df_people_by_time = df_people_by_time.reset_index(drop=True)
#         df_people_by_time = df_people_by_time.drop(['sum'],axis=1)
    else:
        df_people_by_time = df_people_by_time.sort_values(ascending=False, by=time)
        df_people_by_time = df_people_by_time.reset_index(drop=True)

    emails_sent = df_people_by_time[time].values[:top]
    plt.bar(np.arange(top),height = emails_sent)
    plt.ylim(0,emails_sent.max()+20)

    # choose different labels
    if show_label == 'first':
        label = []
        for i in range(top):
            label.append(df_people_by_time['people'].values[:top][i][0])
    elif show_label == 'last':
        label = []
        for i in range(top):
            label.append(df_people_by_time['people'].values[:top][i][1])
    elif show_label == 'email':
        label = []
        for i in range(top):
            label.append(df_people_by_time['people'].values[:top][i][2])
    elif show_label == 'name':
        label = []
        for i in range(top):
            label.append(df_people_by_time['people'].values[:top][i][:2])
    else:
        label = df_people_by_time['people'].values[:top]
    plt.xticks(np.arange(top),label,rotation=90)
    if save_to_file!=None:
        plt.savefig(save_to_file+'.pdf')

#----------------------------------------------------------------------
def standardize_triplet(l ,to_type=tuple):
    if type(l[0][0]) == str:
        for i in range(len(l)):
            l[i] = to_type(l[i])
    else:
        for i in range(len(l)):
            for j in range(len(l[i])):
                l[i][j] = to_type(l[i][j])
    return l
