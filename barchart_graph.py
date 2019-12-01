import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from function_library import *

l_from = fromPickle('from_list')
l_sent = fromPickle('sent_list')

unique_senders = list(uniqueEmails(l_from))
unique_senders.sort()

name2id, id2name = nameToIndexDict(unique_senders)
createSenderMatrix(l_sent, l_from, unique_senders, name2id, save=True)
