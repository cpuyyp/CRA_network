# 2019-11-30
# Author: Joey Jingze and Gordon Erlebacher
# GE: refactor code to prepare for more general usage. 
#   Add functions, and eventually make it class-based

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import embed
from function_library import *

#def dictFreq(in_dict):

d_final = fromPickle("d_final")

#----------------------------------------------------------------------
# create a square matrix to save number of emails sent and received by each person

# do not copy index onto first column: index=False
# Assumes: df.to_csv("clean_output_noindex.csv", index=False)
# New index starts from 0, consecutively
l_to   = fromPickle("to_list")
l_from = fromPickle("from_list")
l_cc   = fromPickle("cc_list")

l_emails = []
for k,v in d_final.items():
   l_emails.append(v[2])
assert(len(l_emails) == len(d_final))

d_pairs = sendReceiveList(l_from, l_to, l_cc, 40, 40)

#----------------------------------------------------------------------
unique_names = l_emails  # could be a list of names for different effects 
name2id, id2name = nameToIndexDict(l_emails)
# s2r is Sender to(2) recipient
s2r = createConnectionMatrix(unique_names, name2id)
nb_mails = np.sum(s2r)  
print("nb mails= ", nb_mails)
embed(); 

#----------------------------------------------------------------------

# plotting
G = nx.Graph()
edge_width = []
node_weight_sender = np.sum(s2r, axis = 1)
node_weight_recipient = np.sum(s2r,axis = 0)
node_weight_total = node_weight_sender + node_weight_recipient 

#nb_recipient = np.array(send_to_recipient.shape[0])
 
# node_weight is the size of node 
# the node weight has to be in a specific order(in the order of time when the node first added to the graph), 
# cannot just use node_weight_total
node_weight = [] 
for i in range(s2r.shape[0]):
    for j in range(i,s2r.shape[0]):
        # if there is more than 1 email between these 2 people, add node if haven't add. Add edge.
        if s2r[i,j] + s2r[j,i] > 1:
            if id2name[i] not in G.nodes():
                G.add_node(id2name[i])
                node_weight.append(node_weight_total[i])
            if id2name[j] not in G.nodes():
                    G.add_node(id2name[j])
                    node_weight.append(node_weight_total[j])
            # To disable edges temporarily, comment out the next line. The spring will not take them into account
            G.add_edge(id2name[i], id2name[j],weight= 2/(s2r[i,j] + s2r[j,i] + 0.5*(node_weight_total[i]+ node_weight_total[j])))
            edge_width.append(s2r[i,j] + s2r[j,i])
print('done adding edges and nodes')


# find who should be labeled
node_have_label = {}
for i in range(s2r.shape[0]):
    if node_weight_total[i]>250 and id2name[i] in G.nodes():
        node_have_label[id2name[i]] = id2name[i]

# edge_width is actrually edge strength. Bigger strength will lead to shorter distance
# edge_width = np.sqrt(np.array(edge_width))
edge_width = 0.2*(np.array(edge_width))
print('done adding labels')

plt.figure(figsize=(40,40))
# calculating node positions
pos = nx.spring_layout(G,iterations=30)
print('done calculating position')

nx.draw_networkx_nodes(G, pos, node_size= node_weight,node_color = 'black')
#nx.draw_networkx_edges(G, pos, width= edge_width, edge_color = 'grey')
nx.draw_networkx_labels(G, pos, labels= node_have_label, font_size=24, font_color = 'red', font_family='sans-serif')

plt.axis('off')
#plt.show()
plt.savefig("gordon.pdf")

