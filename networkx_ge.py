
# coding: utf-8

# JOEY

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle

"""
string = r"gordon (is) wonderful (again) gone"
print(string)
string = re.sub(r"\(.*\))", "", string) 
print(string)
quit()
middle = re.sub(r"\.", "", "k.")
print("middle= ", middle)
quit()
string = r"Gordon Frank A"
stri = re.split(r"[, ]", string)
print(stri)
quit()
"""


# In[2]:

def readDataFrame(file_name):
    df = pd.read_csv(file_name, index_col=False)
    #print(df.head())
    #print(df.columns)
    cols = df.columns
    df = df.drop(columns=[cols[0], cols[1]]) 
    #print("dropped")
    #print(df.head())
    df['Sent'] = pd.to_datetime(df['Sent'])
    df = df[df['Sent'] > datetime(1900, 1, 1, 0, 0, 0)]
    df = df.reset_index(drop=True)
    #print("df= ", df)
    #print("len(df)= ", df.columns())
    #del df['Unnamed: 0.1']
    return df



def restrictEmailsToYears(df, low=2013, high=2014):
    # restrict emails
    df = df[df['Sent'] > datetime( low, 1, 1, 0, 0, 0)]
    #print("df1: ", df.head())
    df = df[df['Sent'] < datetime(high, 1, 1, 0, 0, 0)]
    #print("df2: ", df.head())
    return df



# In[4]:


# testing for name correction
name_tests = ['Administration-Thomas Harrison <Abc.abc@gmail.com>',
'Wiebler, Brian T.',
'Brian T. Wiebler',
'JT Burnette (jt@inkbridge.com)',
'JTBurnette (jt@inkbridge.com)',
'Ingram, M\'Lisa',
'Mike V',
'LCEM Mail',
'Gary Yordon [mailto:gary@govinc.net]',
'City Commission Office',
'(gary@govinc.net)<gary@govinc.net>',
'alan1596@aol.com',
' april salter']
#re_name1 = re.compile(r'.*?([A-Z][a-z]+)\s?[A-Z]?\.?\s?([A-Z][a-z]+\s?[A-Z]?[a-z]+)')
#re_name2 = re.compile(r'.*?([A-Z][a-z]+),\s?[A-Z]?\'?([A-Z][a-z]+)\s?[A-Z]?\.?')
#re_name3 = re.compile(r'.*?([A-Z]+?)\s?([A-Z]?[a-z]+)\s?[\[\(]?.*\@.*[\)\]]?')

# Remove "?([mailto:.*])" (shortest string)
# Remove "?(<i.*>)" (shortest string)
# Identify commas:   Last, First Middle
# orig
#re_name1 = re.compile(r'([A-Z][a-z]+)\s?[A-Z]?\.?\s?([A-Z][a-z]+\s?[A-Z]?[a-z]+)')
#re_name2 = re.compile(r'([A-Z][a-z]+),\s?[A-Z]?\'?([A-Z][a-z]+)\s?[A-Z]?\.?')
#re_name3 = re.compile(r'([A-Z]+?)\s?([A-Z]?[a-z]+)\s?[\[\(]?.*\@.*[\)\]]?')

#re_email_orig = re.compile(r'.*?([a-zA-Z_]*\.?\w+@[0-9a-zA-Z_-]*\.?[a-zA-Z_]*\.?[a-zA-Z]{2,3})') # works
# The following defition allows double dots in the emai. But since we are scanning emails that have 
# been sent, we will ignore this issue. 
# WHY the ".*?" at the beginning of the string. That makes the match start at the start of the strong
#re_email = re.compile(r'.*?([a-zA-Z_.]*\.?\w+@[0-9a-zA-Z._-]*)')
re_email = re.compile(r'([a-zA-Z_.]*\.?\w+@[0-9a-zA-Z._-]*)')
#orig re_email = re.compile(r'.*?([a-zA-Z_]*\.?\w+@[a-zA-Z_]*\.?[a-zA-Z_]*\.?[a-zA-Z]{2,3})')

def extract_email(string):
    # Only keep first email found, and remove it from the string
    print("orig string= ", string)
    s = re_email.search(string)
    #print("search: ", s)
    #print("dir(s): ", dir(s))
    #m = re_email.match(string)
    #    m = re.match(re_email, string)
    #print("match: ", m)
    ##print("m.groups= ", m.groups())
    #print("dir(m): ", dir(m))
    if s:
        #print("search.group(0): ", s.group(0))
        #print("match.group(0): ", m.group(0))
        #print("findall: ", re_email.findall(string))
        email = re_email.findall(string)[0]
        #print("email= ", email)
        #print("start, end: ", s.start(), s.end())
        string = string[:s.start()] + string[s.end():]
        string = re.sub(email, "", string) 
        string = re.sub(r"<>", "", string) 
        string = re.sub(r"\[mailto:\]", "", string) 
        #print("removing junk: string= ", string)
        string = re.sub(r"\(\)", "", string) 
        #print("after removal: ", string)
        string = string.strip()
        # Takes shortest pattern unless enclosed in grouping ()
    else:
        email = None
    #print("before sub: string= ", string)
    pattern = r"on behalf.*$"
    string = re.sub(pattern, "", string.lower())
    #print("after sub: string= ", string)
    #print("new string= ", string)
    return email, string


def standardize_name2(string): 
# The emails have already been removed. All that is left is a single person's name
    print("standardize string: ", string)
    string = string.strip()
    if len(string) == 0:
        return '', '', ''

    # remove commas and dots
    string = re.sub(r"[\.\"]", "", string)

    strings = string.split(',')
    first = last = middle = ''

    if len(strings) == 1:
        strings1 = strings[0].split(' ')
        lg = len(strings1)
        # remove empty strings
        if lg == 1:
            pass
        elif lg == 2:
            first = strings1[0]
            middle = ''
            last = strings1[1]
        elif lg > 2:
            first = strings1[0]
            middle = " ".join(strings1[1:-1])
            middle = re.sub(r"\.", "", middle)
            last  = strings1[-1]

    elif len(strings) == 2:
        last = strings[0]
        first = strings[1].strip()  # could contain middle initial
        first = re.sub(r"\.", "", first) # remove dots
        splits = first.split(" ")
        first = splits[0]
        middle = " ".join(splits[1:])

    elif len(strings) > 2:
        print("length(strings)>2, should not happen: ", strings)
        #quit()

    return first.strip(), middle.strip(), last.strip()



df = readDataFrame("output4.csv")
df = restrictEmailsToYears(df, low=2013, high=2014)
# In[5]:

# convert pandas df to dictionary, only keep sender/recipient names and sent time
email_dic = {}
max_nb_recipients = 10
max_nb_cc = 10

FROM = 1 
SENT = 2
TO   = 3 
CC   = 4 
SUBJ = 5 
ATTCH= 6 
BODY = 7 
#print(df.iloc[:,:])
print(df.columns)
# Index(['From', 'Sent', 'To', 'CC', 'Subject', 'Attachments', 'Body'], dtype='object')

# Clean the dataframe
# Replace all senders by their standardized names
# Replace all TO: by their standardized names
# Replace all CC: by their standardized names

#def createNameDict(df):
    #names = df['

#def createEmailDict(df):

to = df['To']
cc = df['CC']
sender = df['From']

#----------------------------------------------------------------------
def cleanDFColumn(df, col_name ):
    recipients = []
    for rec in df[col_name]:
        recipient_string = rec[2:-2].replace("'",'')
        if re.search("tosteen@moorebass.com", recipient_string):
            flag = True
        else: 
            flag = False
        r1 = recipient_string.split(';')

        if len(r1) > 1:
            recipient_list = r1
        elif len(r1) == 1:
           r2 = recipient_string.split(',') 
           if len(r2) == 1:
              recipient_list = r2
              # no further splits necessary. Single name or single email. Pass to standardize_name2()
           else:  # If there are double quotes, remove commas from double quotes, 
               # if the recipient list is all letters, spaces, "-", "." and commas, 
               # then we are dealing with a single user name.
               # pattern = r"[A-Za-z\._-,\s]+" 
               # match entire string with letters, comma, "-"
               pattern = r"(^[A-Za-z, \.]+$)"
               #print("recipient_list= ", recipient_list)
               m = re.search(pattern, recipient_string)
               if m:  # person's name
                   recipient_list = [recipient_string]
               else:
                   print("-----------------------------------------")
                   #print("recipient_string= ", recipient_string)
                   #pattern = r"(\b\w+\b),(\b\w_\b)"
                   pattern = r"\s*\w+\s*,\s*\w+\s*"
                   m = re.search(pattern, recipient_string)
                   if not m:
                       recipient_list = recipient_string.split(',') 
                       #print("recipient_list= ", recipient_list)
                   else:
                       nb_commas = len(re.findall(",", recipient_string))
                       print("nb commas= ", nb_commas)
                       if nb_commas > 2:  # not foolproof. There might be only two emails. 
                           recipient_list = recipient_string.split(',')
                       else:
                           recipient_list = [recipient_string]
                       # Separate by commas
        #-------------------

        # At this stage, recipient_list is the list of recipients
        # to be further processed by extract_mail and standardize_name2

        for i, person in enumerate(recipient_list):
            #print("person= ", person)
            email, new_str = extract_email(person)
            #print("email= ", email, ",  new_str= ", new_str)
            f, m, l = standardize_name2(new_str)
            recipient_list[i] = (f,l,email)
            #print("f,m,l= ", [f,m,l])
        recipients.append(recipient_list)
        #if flag: quit()
    df[col_name] = recipients
    return df

#----------------------------------------------------------------------

def cleanSenders(df):
    senders = []
    for rec in df['From']:
        sender = rec[1:-1].replace("'",'')
        email, new_str = extract_email(sender)
        f, m, l = standardize_name2(new_str)
        # strip() should not be necessary, but the line with "sheila" has an additional leading space
        # I do not know why. 
        senders.append((f,l,email))  # ignore the middle initials
    df[FROM] = senders

cleanDFColumn(df, 'To')
cleanDFColumn(df, 'CC')
#cleanSenders(df)

s = set()
print(df.columns)
to_list = df['To'].values
#------------------
print("==================================")
print("== Column 'To' =====================")
for i, lst in enumerate(to_list):
   #rec = df[TO].iloc[i].values
   print("--- i= ", i, " ------")
   for r in lst:
       print("to:    ", r)
   #print(df[TO].iloc[i])
   #s.add(df[TO].iloc[i])
#------------------
print("==================================")
print("== Column 'CC' =====================")
for i, lst in enumerate(to_list):
   print("--- i= ", i, " ------")
   for r in lst:
       print("cc:    ", r)
quit()

#print("len(df[FROM])= ", len(df[TO].values))
print("len(set)= ", len(s))

quit()


# compute email_dict: 
#for idx, dataslice in enumerate(df.itertuples()):
recipients = []
print("----------------------------------------")
for i, row in enumerate(df.itertuples()):
    print
    standardized_name = []
#   print(dataslice[4])
    #recipient_list = dataslice[4][2:-2].replace("'",'').strip().split(';')
    recipient_list = row[TO][2:-2].replace("'",'').strip().split(';')
    if i < 3:
        print("i= ", i)
        print(row)
        print("recipient_list= ", recipient_list)
        print("row[TO] = ", row[TO])
        print("row[CC] = ", row[CC])
    recipients.append(recipient_list)

    continue

    for person in recipient_list:
#         print(person)
        if len(recipient_list) > max_nb_recipients: break
        standardized_name.append(standardize_name(person))
#         print(standardize_name(person))

    #cc_list = dataslice[5][2:-2].replace("'",'').strip().split(';')
    cc_list = row[CC][2:-2].replace("'",'').strip().split(';')
    for person in cc_list:
        if len(cc_list) > max_nb_cc: break
        if person != '':
            standardized_name.append(standardize_name(person))

    for name in standardized_name:
        if standardize_name(row[FROM]) not in email_dic:
            email_dic[standardize_name(row[FROM])] = []
        if standardize_name(row[FROM]) != None:
            if name != None:
                #email_dic[standardize_name(dataslice[2])].append([name,dataslice[3]])
                email_dic[standardize_name(row[FROM])].append([name,row[SENT]])

#print(email_dic)
#print(recipients)
print("----------------------------------------")
print("recipients[0:3]= ", recipients[0:3])
print("df[TO][0:3]= ", df[TO])
quit()
df[TO] = recipients
print(df.iloc[0:3,TO])
quit()

def saveData(file_name, email_dic):
    with open('email_dic.pickle', 'wb') as handle:
        pickle.dump(email_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

saveData('email_dic.pickle', email_dic)

# In[6]:


# the key of the dictionary is sender's name, the value coresponds to the key is the recipient's name and sent time.
def loadData(file_name):
    with open('email_dic.pickle', 'rb') as handle:
        email_dic = pickle.load(handle)
    return  email_dic.copy()

email_dic_c = loadData('email_dic.pickle')
#quit()
#----------------------------------------------------------------------


# In[10]:

nb_senders = len(email_dic_c.keys())
print("There are %d distinct senders" % nb_senders)

all_names = []
for sender in email_dic_c.keys():
#     print(sender)
    if sender != None:
        all_names.append(sender)
    for e in email_dic_c[sender]:
        recipient = e[0]
        if recipient != None:
            all_names.append(recipient)


for idx in range(len(all_names)):
    all_names[idx] = all_names[idx].replace("'",'').strip()
    all_names[idx] = standardize_name(all_names[idx])
unique_names = list(set(all_names))
print("nb unique names: ", len(unique_names)) #659
print("nb all_names: ", len(all_names))  # 6311


# In[11]:


#print(type(unique_names))
#print(dir(unique_names))
#print(unique_names)
unique_names.remove(None)
unique_names.sort()
unique_names[-100:]

print("remove None from unique_names")
print("nb unique_names: ", len(unique_names))  #658
#quit()



# In[12]:


# create name to index dictionary and index to name dictionary for later use
name_id = {}
for idx,name in enumerate(unique_names):
    name_id[name] = idx
id_name = {}
for idx,name in enumerate(unique_names):
    id_name[idx] = name


# In[13]:


# create a square matrix to save number of emails sent and received by each person
sender_to_recipient = np.zeros((len(unique_names),len(unique_names)))
for sender in email_dic_c.keys():
    for e in email_dic_c[sender]:
        recipient = e[0]
        if sender in name_id and recipient in name_id:
            sender_to_recipient[name_id[sender],name_id[recipient]] = sender_to_recipient[name_id[sender],name_id[recipient]] +1
np.sum(sender_to_recipient)


# In[14]:


# plotting
G = nx.Graph()
edge_width = []
node_weight_sender = np.sum(sender_to_recipient,axis = 1)
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



# # old codes

# In[ ]:


# # for i in range(5):
# i = 1
# df_for_plot2 = df_for_plot[df_for_plot['Sent']<datetime(2013+i, 1, 1, 0, 0, 0)]
# df_for_plot3 = df_for_plot2[df_for_plot2['Sent']>datetime(2012+i, 1, 1, 0, 0, 0)]

# G = nx.from_pandas_edgelist(df_for_plot3, 'From','To')
# count = df_for_plot3['From'].append(df_for_plot3['To']).value_counts()
# nodesizes = np.zeros(len(list(G.nodes)))
# for i in range(len(list(G.nodes))):
#     if list(G.nodes)[i] == None:
#         nodesizes[i] = 0
#     else:
#         nodesizes[i] = count[count.index == list(G.nodes)[i]][0]
# index = nodesizes.argsort()[-10:][::-1]
# Top10 = [list(G.nodes)[i] for i in index]
# Top10_val = [nodesizes[i] for i in index]
# print(list(zip(Top10, Top10_val)))

# plt.figure(figsize = (40,30))
# pos = nx.spring_layout(G, k = 0.1, iterations = 30)
# nx.draw_networkx(G, pos, node_size = nodesizes, node_color = 'black', with_labels = True, edge_color='grey')
# plt.axis('off')
# plt.show()


# In[ ]:


# l1 = list(G.nodes())
# l2 = big_names
# l3 = [x for x in l1 if x not in l2]
# G.remove_nodes_from(l3)


# In[ ]:


# count = df_for_plot3['From'].append(df_for_plot3['To']).value_counts()
# nodesizes = np.zeros(len(list(G.nodes)))
# for i in range(len(list(G.nodes))):
#     if list(G.nodes)[i] == None:
#         nodesizes[i] = 0
#     else:
#         nodesizes[i] = count[count.index == list(G.nodes)[i]][0]
# index = nodesizes.argsort()[-10:][::-1]
# Top10 = [list(G.nodes)[i] for i in index]
# Top10_val = [nodesizes[i] for i in index]
# print(list(zip(Top10, Top10_val)))

# plt.figure(figsize = (40,30))
# pos = nx.spring_layout(G, k = 0.1, iterations = 10)
# nx.draw_networkx(G, pos, node_size = nodesizes, node_color = 'black', with_labels = True, edge_color='grey')
# plt.axis('off')
# plt.savefig('nx_200.png')
# plt.show()

