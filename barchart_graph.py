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

l_to   = fromPickle("to_list")
l_from = fromPickle("from_list")
l_cc   = fromPickle("cc_list")

l_to = standardize_triplet(l_to)
l_from = standardize_triplet(l_from)
l_cc = standardize_triplet(l_cc)

l_year = fromPickle('year_list')
l_month = fromPickle('month_list')
l_week = fromPickle('week_list')
l_weekday = fromPickle('weekday_list')
l_hour = fromPickle('hour_list')


start_time = 2012
end_time = 2018
df_people_by_time = createPeopleMatrix(l_from, l_year, start_time, end_time)
# df_people_by_time = createPeopleMatrix(l_from, l_month, 0, 12)

nrows = 2
ncols = 3
#plt.subplots(2,3,figsize=(12,10))
plt.subplots(nrows,ncols,figsize=(12,10))

nb_top = 50
plt.rcParams.update({'font.size':6})
plt.suptitle('#Emails sent by top %d'%nb_top, y=1.02, fontsize = 20)

for time in range(start_time,end_time):
# for year in range(0,12):
    #plt.subplot(2,3,year-2011)
    plt.subplot(nrows,ncols,time-start_time+1)
    plt.gca().set_title('Time {}'.format(time))
    plot_barchart_by_time(df_people_by_time, time, top = nb_top, sortby = 'total', show_label = 'last', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot.pdf', bbox_inches='tight')

# generating stacked bar chart
plt.figure(figsize=(12,10))
plot_stacked_barchart(df_people_by_time, top = nb_top, normalize = False, sortby = 'total', show_label = 'first', remove_blank = True, save_to_file='stacked_barplot')

# plotting connection matrix
unique_people = set()
for i in range(len(l_from)):
    unique_people.add(l_from[i])

for i in range(len(l_cc)):
    for lst in l_cc[i]:
        unique_people.add(lst)

for i in range(len(l_to)):
    for lst in l_to[i]:
        unique_people.add(lst)
unique_people = list(unique_people)
unique_people.sort()
name2id, id2name = nameToIndexDict(unique_people)
s_to_r = createConnectionMatrix(unique_people, name2id, l_from=l_from, l_to=l_to, l_cc=l_cc)
plot_connection_matrix(s_to_r,unique_people, top = 30, show_label='last', remove_blank = True, save_to_file='connection_matrix')
