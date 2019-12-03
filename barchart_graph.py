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

#df_people_by_time = createPeopleMatrix(l_from, l_year, 2012, 2018)
df_people_by_time = createPeopleMatrix(l_from, l_month, 0, 12)

nrows = 4
ncols = 4
#plt.subplots(2,3,figsize=(12,10))
plt.subplots(4,3,figsize=(12,10))

nb_top = 50
plt.rcParams.update({'font.size':4})
plt.suptitle('#Emails sent by top %d'%nb_top,y=1.02,fontsize = 20)

#for year in range(2012,2018):
for year in range(0,12):
    #plt.subplot(2,3,year-2011)
    plt.subplot(4,3,year+1)
    plt.gca().set_title('Year {}'.format(year))
    plot_barchart_by_time(df_people_by_time, year, top = nb_top, sortby = 'total', show_label = 'last', remove_blank = True)
plt.tight_layout()
plt.savefig('barplot.pdf', bbox_inches='tight')
