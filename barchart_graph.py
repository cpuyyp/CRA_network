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
l_cc = fromPickle('cc_list')
l_to = fromPickle('to_list')


l_year = fromPickle('year_list')
l_month = fromPickle('month_list')
l_week = fromPickle('week_list')
l_weekday = fromPickle('weekday_list')
l_hour = fromPickle('hour_list')


df_people_by_time = createPeopleMatrix(l_cc, l_year, 2012, 2018)

plot_barchart_by_time(df_people_by_time, 2014, top = 20, sortby = 'total', remove_blank = True, save_to_file = 'top20 in 2014')
