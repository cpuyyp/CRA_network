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

nb_top = 25
plt.rcParams.update({'font.size':12})

# plot barplot 1
nrows = 2
ncols = 3
plt.subplots(nrows,ncols,figsize=(12,8))
plt.suptitle('#Emails sent by top %d people'%nb_top, y=1.02, fontsize = 20)
for time in range(start_time,end_time):
    plt.subplot(nrows,ncols,time-start_time+1)
    plt.gca().set_title('Year {}'.format(time))
    plot_barchart_by_time(df_people_by_time, time, top = nb_top, sortby = 'total', show_label = 'name', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot.pdf', bbox_inches='tight')
plt.savefig('barplot.png', bbox_inches='tight')

# plot barplot 2
l_people = [ [a]+b+c for a,b,c in zip(l_from,l_to,l_cc)]
df_people_by_time = createPeopleMatrix(l_people, l_year, start_time, end_time)
plt.subplots(nrows,ncols,figsize=(12,8))
plt.suptitle('#Emails involved by top %d people'%nb_top, y=1.02, fontsize = 20)
for time in range(start_time,end_time):
    plt.subplot(nrows,ncols,time-start_time+1)
    plt.gca().set_title('Year {}'.format(time))
    plot_barchart_by_time(df_people_by_time, time, top = nb_top, sortby = True, show_label = 'name', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot2.pdf', bbox_inches='tight')
plt.savefig('barplot2.png', bbox_inches='tight')

# plot barplot 3
start_time = 1
end_time = 13
df_people_by_time = createPeopleMatrix(l_from, l_month, start_time, end_time)
nrows = 4
ncols = 3
plt.subplots(nrows,ncols,figsize=(12,16))
plt.suptitle('#Emails sent in each month by top %d people'%nb_top, y=1.02, fontsize = 20)
for time in range(start_time,end_time):
    plt.subplot(nrows,ncols,time-start_time+1)
    plt.gca().set_title('Month {}'.format(time))
    plot_barchart_by_time(df_people_by_time, time, top = nb_top, sortby = 'total', show_label = 'name', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot3.pdf', bbox_inches='tight')
plt.savefig('barplot3.png', bbox_inches='tight')

# generating stacked bar chart 1
nb_top = 50
start_time = 2012
end_time = 2018
df_people_by_time = createPeopleMatrix(l_from, l_year, start_time, end_time)
plt.figure(figsize=(10,6))
plt.title('#Emails sent in total by top %d people'%nb_top, y=1.02, fontsize = 20)
plot_stacked_barchart(df_people_by_time, top = nb_top, normalize = False, sortby = 'total', show_label = 'name', remove_blank = True, save_to_file='stacked_barplot')

# generating stacked bar chart 2
nb_top = 20
start_time = 0
end_time = 6
df_people_by_time = createPeopleMatrix(l_from, l_hour, start_time, end_time)
plt.figure(figsize=(6,5))
plt.title('#Emails sent at midnight by top %d people'%nb_top, y=1.02, fontsize = 20)
plot_stacked_barchart(df_people_by_time, top = nb_top, normalize = False, sortby = 'total', show_label = 'name', remove_blank = True, save_to_file='stacked_barplot2')

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
plot_connection_matrix(s_to_r,unique_people, top = 30, show_label='name', remove_blank = True, save_to_file='connection_matrix')


# select a specific year and get monthly plot
ind = np.array(l_year) == 2013
l_from_selected = np.array(l_from)[ind].tolist()
l_month_selected = np.array(l_month)[ind].tolist()
l_from_selected = standardize_triplet(l_from_selected)
df_people_by_time = createPeopleMatrix(l_from_selected, l_month_selected, 1, 13)
plt.subplots(4,3,figsize=(10,12))
plt.suptitle('# email sent by the top 20 people in 2013 per year',y=1.02,fontsize = 20)
for time in range(1,13):
    plt.subplot(4,3,time)
    plt.gca().set_title('Month {} in 2014'.format(time))
    plot_barchart_by_time(df_people_by_time, time, top = 20, sortby = 'total', show_label = 'name', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot_monthly.pdf', bbox_inches='tight')
plt.savefig('barplot_monthly.png', bbox_inches='tight')
