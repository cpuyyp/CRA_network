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

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# create a square matrix to save number of emails sent and received by each person
def emailsSentReceivedPerPerson(unique_names, email_dic_c):
    sender_to_recipient = np.zeros((len(unique_names),len(unique_names)))
    for sender in email_dic_c.keys():
        for e in email_dic_c[sender]:
            recipient = e[0]
            if sender in name_id and recipient in name_id:
                sender_to_recipient[name_id[sender],name_id[recipient]] = sender_to_recipient[name_id[sender],name_id[recipient]] +1
    np.sum(sender_to_recipient)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# name2id is better name than name_id

# do not copy index onto first column: index=False
# Assumes: df.to_csv("clean_output_noindex.csv", index=False)
# New index starts from 0, consecutively

l_to   = fromPickle("to_list")
l_from = fromPickle("from_list")
l_cc   = fromPickle("cc_list")

d_pairs = sendReceiveList(l_from, l_to, l_cc, 40, 40)

#----------------------------------------------------------------------
# name2id is better name than name_id
unique_names = ['a', 'b']
d_email = {}
name2id, id2name = nameToIndexDict(unique_names)
emailsSentReceivedPerPerson(unique_names, d_email)
embed(); quit()



# plotting
G = nx.Graph()
edge_width = []
node_weight_sender = np.sum(sender_to_recipient, axis = 1)
node_weight_recipient = np.sum(sender_to_recipient,axis = 0)
node_weight_total = node_weight_sender + node_weight_recipient 

#nb_recipient = np.array(send_to_recipient.shape[0])
 
# node_weight is the size of node 
# the node weight has to be in a specific order(in the order of time when the node first added to the graph), 
# cannot just use node_weight_total
node_weight = [] 
for i in range(sender_to_recipient.shape[0]):
    for j in range(i,sender_to_recipient.shape[0]):
        # if there is more than 1 email between these 2 people, add node if haven't add. Add edge.
        if sender_to_recipient[i,j] + sender_to_recipient[j,i] > 1:
            if id_name[i] not in G.nodes():
                G.add_node(id_name[i])
                node_weight.append(node_weight_total[i])
            if id_name[j] not in G.nodes():
                    G.add_node(id_name[j])
                    node_weight.append(node_weight_total[j])
            G.add_edge(id_name[i], id_name[j],weight= 2/(sender_to_recipient[i,j] + sender_to_recipient[j,i] + 0.5*(node_weight_total[i]+ node_weight_total[j])))
            edge_width.append(sender_to_recipient[i,j] + sender_to_recipient[j,i])
print('done adding edges and nodes')


# find who should be labeled
node_have_label = {}
for i in range(sender_to_recipient.shape[0]):
    if node_weight_total[i]>250 and id_name[i] in G.nodes():
        node_have_label[id_name[i]] = id_name[i]

# edge_width is actrually edge strength. Bigger strength will lead to shorter distance
# edge_width = np.sqrt(np.array(edge_width))
edge_width = 0.2*(np.array(edge_width))
print('done adding labels')

plt.figure(figsize=(40,40))
# calculating node positions
pos = nx.spring_layout(G,iterations=30)
print('done calculating position')

nx.draw_networkx_nodes(G, pos, node_size= node_weight,node_color = 'black')
nx.draw_networkx_edges(G, pos, width= edge_width, edge_color = 'grey')
nx.draw_networkx_labels(G, pos, labels= node_have_label, font_size=24, font_color = 'red', font_family='sans-serif')

plt.axis('off')
#plt.show()
plt.savefig("gordon.pdf")


# In[15]:


# test
edge_width.max()


# In[16]:


# test
G = nx.Graph()

G.add_edge('d','a',weight = 0.1)
G.add_edge('d','b',weight = 100)
G.add_edge('c','a',weight = 0.1)

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos,node_color = 'black')
nx.draw_networkx_edges(G, pos, width=1, edge_color = 'grey')
nx.draw_networkx_labels(G, pos, labels={'a':'a','d':'d'}, font_size=30, font_color='blue',font_family='sans-serif')


