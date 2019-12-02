import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from function_library import *
from function_library2 import *

l_from = fromPickle('from_list')
l_sent = fromPickle('sent_list')

unique_senders = list(uniqueEmails(l_from))
unique_senders.sort()

name2id, id2name = nameToIndexDict(unique_senders)
df_sender_by_year = createSenderMatrix(l_sent, l_from, unique_senders, name2id, save=True)

plt.figure(figsize = (30,20))
for i in range(6):
    plt.subplot(3,2,i+1)
    emails_sent_in_a_year = df_sender_by_year[i+2012].values
    plt.bar(np.arange(len(unique_senders)),height = emails_sent_in_a_year)
    plt.ylim(0,emails_sent_in_a_year.max())
plt.savefig('barplot.pdf')
