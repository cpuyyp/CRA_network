
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
    # l_people is a list of triplets or a list of list triplets
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
    # If sortby
    #    'total',  the total number of emails in all years
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
            first_name= df_people_by_time['people'].values[:top][i][0]
            last_name= df_people_by_time['people'].values[:top][i][1]
            label.append(first_name + ' '+ last_name)
    else:
        label = df_people_by_time['people'].values[:top]
    plt.xticks(np.arange(top),label,rotation=90)

    if save_to_file!=None:
        plt.savefig(save_to_file+'.pdf')
        plt.savefig(save_to_file+'.png')

#----------------------------------------------------------------------
# GE: What does this function do?
# GE: The less loops you use, the faster the code
# GE: describe the arguments. What is l? What type? What do subscripts mean?
def standardize_triplet(l, to_type=tuple):
    try:
        if type(l[0][0]) == str:
            for i in range(len(l)):
                l[i] = to_type(l[i])
        else:
            for i in range(len(l)):
                for j in range(len(l[i])):
                    l[i][j] = to_type(l[i][j])
    except:
        pass
    return l

#----------------------------------------------------------------------
def plot_stacked_barchart(df_people_by_time, top = 20, normalize = True, sortby = 'total', show_label = 'first', remove_blank = True, save_to_file=None):
    df = df_people_by_time.copy()
    if remove_blank == True:
        index = df.index[df['people'] == ('', '', '')]
        df = df.drop(index)
        df = df.reset_index(drop=True)
    if sortby == 'total':
        df['sum'] = df.sum(axis=1)
        df = df.sort_values(ascending=False, by='sum')
        df = df.reset_index(drop=True)
#         df = df.drop(['sum'],axis=1)
    cols = df.columns
    if normalize == True:
        for col in cols[1:-1]:
            df[col] = df[col]/df['sum']
    if top == None:
        top = df.values.shape[0]
    bottom = 0
    for col in cols[1:-1]:
        emails_sent = df[col].values[:top]
        plt.bar(np.arange(top), height = emails_sent, bottom = bottom, alpha=0.8,label = str(col)+'-'+str(col+1))
        bottom = bottom + emails_sent

    # choose different labels
    if show_label == 'first':
        label = []
        for i in range(top):
            label.append(df['people'].values[:top][i][0])
    elif show_label == 'last':
        label = []
        for i in range(top):
            label.append(df['people'].values[:top][i][1])
    elif show_label == 'email':
        label = []
        for i in range(top):
            label.append(df['people'].values[:top][i][2])
    elif show_label == 'name':
        label = []
        for i in range(top):
            first_name= df['people'].values[:top][i][0]
            last_name= df['people'].values[:top][i][1]
            label.append(first_name + ' '+ last_name)
    elif show_label == None:
        pass
    else:
        label = df['people'].values[:top]
    if show_label != None:
        plt.xticks(np.arange(top),label,rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.ylim(0,1.1*bottom.max())
    if save_to_file!=None:
        plt.savefig(save_to_file+'.pdf')
        plt.savefig(save_to_file+'.png')

#----------------------------------------------------------------------
def plot_connection_matrix(s2r,unique_people, sort = False, top = 30, show_label='first', figsize=(12,10), remove_blank = True, save_to_file=None):
    s_to_r = s2r.copy()
    if sort == True:
        row_sum = s_to_r.sum(axis = 1)
        col_sum = s_to_r.sum(axis = 0)
        row_ind = np.argsort(row_sum)[::-1]
        col_ind = np.argsort(col_sum)[::-1]
        s_to_r = s_to_r[row_ind,:]
        s_to_r = s_to_r[:,col_ind]
        y_triplet = np.array(unique_people)[row_ind]
        x_triplet = np.array(unique_people)[col_ind]
    else:
        y_triplet = np.array(unique_people)
        x_triplet = np.array(unique_people)
    if remove_blank == True:
        x_index = np.where(x_triplet == ['', '', ''])[0][0]
        y_index = np.where(y_triplet == ['', '', ''])[0][0]
        x_triplet = np.delete(x_triplet, x_index, axis=0)
        y_triplet = np.delete(y_triplet, y_index, axis=0)
        s_to_r = np.delete(s_to_r, x_index, axis=1)
        s_to_r = np.delete(s_to_r, y_index, axis=0)

    xlabel = []
    ylabel = []
    if show_label == 'first':
        for i in range(top):
            xlabel.append(x_triplet[:top][i][0])
            ylabel.append(y_triplet[:top][i][0])
    elif show_label == 'last':
        for i in range(top):
            xlabel.append(x_triplet[:top][i][1])
            ylabel.append(y_triplet[:top][i][1])
    elif show_label == 'email':
        for i in range(top):
            xlabel.append(x_triplet[:top][i][2])
            ylabel.append(y_triplet[:top][i][2])
    elif show_label == 'name':
        for i in range(top):
            first_name = x_triplet[:top][i][0]
            last_name = x_triplet[:top][i][1]
            xlabel.append(first_name + ' '+ last_name)
            first_name = y_triplet[:top][i][0]
            last_name = y_triplet[:top][i][1]
            ylabel.append(first_name + ' '+ last_name)
    elif show_label == False:
        pass
    else:
        xlabel = x_triplet[:top]
        ylabel = y_triplet[:top]

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.xticks(np.arange(top), xlabel, rotation=90)
    plt.yticks(np.arange(top), ylabel)
    plt.gca().xaxis.set_ticks_position('top')
    plt.imshow(s_to_r[:top,:top],cmap = 'plasma')
    plt.xlabel('recipients',fontsize = 16)
    plt.ylabel('senders',fontsize = 16)
    plt.gca().xaxis.set_label_coords(0.5, 1.2)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('# of emails from sender to recipient', rotation=90, fontsize=16)

    if save_to_file!=None:
        plt.savefig(save_to_file+'.pdf')
        plt.savefig(save_to_file+'.png')

def plot_network(s2r, directed = False, edge_threshold = 0, node_w = 'total', draw_labels = False, label_threshold = 0,iterations = 30, figsize=(40,40)):
    if directed == False:
        G = nx.Graph()
    elif directed == True:
        G = nx.DiGraph()
    else:
        print('directed keyword', directed, 'not defined')

    edge_width = []
    node_weight_sender = np.sum(s2r, axis = 1)
    node_weight_recipient = np.sum(s2r,axis = 0)
    node_weight_total = node_weight_sender + node_weight_recipient
    node_weight = []
    for i in range(s2r.shape[0]):
        for j in range(i+1,s2r.shape[1]):
            # if there is more than n email between these 2 people, add node if haven't add. Add edge.
            if s2r[i,j] + s2r[j,i] > edge_threshold:
                if id2name[i] not in G.nodes():
                    G.add_node(id2name[i])
                    if node_w == 'total':
                        node_weight.append(node_weight_total[i])
                    elif node_w == 'send':
                        node_weight.append(node_weight_sender[i])
                    elif node_w == 'receive':
                        node_weight.append(node_weight_recipient[i])
                    else:
                        print('node_w keyword', node_w, 'not defined')
                if id2name[j] not in G.nodes():
                    G.add_node(id2name[j])
                    if node_w == 'total':
                        node_weight.append(node_weight_total[j])
                    elif node_w == 'send':
                        node_weight.append(node_weight_sender[j])
                    elif node_w == 'receive':
                        node_weight.append(node_weight_recipient[j])
                    else:
                        print('node_w keyword', node_w, 'not defined')
                if directed == False:
                    G.add_edge(id2name[i], id2name[j],weight= 2/(s2r[i,j] + s2r[j,i] + 0.5*(node_weight_total[i]+ node_weight_total[j])))
                elif directed == True:
                    if s2r[i,j] > s2r[j,i]:
                        G.add_edge(id2name[i], id2name[j],weight= 2/(s2r[i,j] + s2r[j,i] + 0.5*(node_weight_total[i]+ node_weight_total[j])))
                    else:
                        G.add_edge(id2name[j], id2name[i],weight= 2/(s2r[i,j] + s2r[j,i] + 0.5*(node_weight_total[i]+ node_weight_total[j])))
                edge_width.append(s2r[i,j] + s2r[j,i])

    edge_width = 0.1*(np.array(edge_width))
    plt.figure(figsize=figsize)

    pos = nx.spring_layout(G,iterations=iterations)

    nx.draw_networkx_nodes(G, pos, node_size= node_weight,node_color = 'black')
    nx.draw_networkx_edges(G, pos, width= edge_width, edge_color = 'grey')
    if draw_labels == True:
        node_have_label = {}
        for i in range(s2r.shape[0]):
            if node_weight_total[i]>label_threshold and id2name[i] in G.nodes():
                if id2name[i][0]+' '+id2name[i][1] != ' ':
                    node_have_label[id2name[i]] = id2name[i][0]+' '+id2name[i][1]
                else:
                    node_have_label[id2name[i]] = 'anonymous'

        nx.draw_networkx_labels(G, pos, labels= node_have_label, font_size=20, font_color = 'red')

    plt.axis('off')
    plt.show()
