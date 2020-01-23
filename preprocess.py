# Preprocessing code. Read in Excel table created by Joey and Austin
# to store the emails from CRA between 2012-2017.
# This code cleans the data further, identifying first, middle, and last names,
# and emails, to remove duplicates.
# Compute the number of unique emails. Define notion of equal users.
# Define the number of different emails for different users.
# Compute whether an email was sent out during daytime hours or after hours [0/1].
# Compute the length of the email, of the subject
# Store all the data in a Pandas file, with the subject and email itself removed to keep
#    the file small
# Consider replacing email names and people's names with indexes for faster execution (for later).

# coding: utf-8

# JOEY, restructured by Gordon Erlebacher, 2019-11-27
# remove all functions


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from function_library import *

# Ideally, have a dictionary of all the compiled searches for efficient
re_email = re.compile(r'([0-9a-zA-Z_\.]*\.?\w+@[0-9a-zA-Z._-]*)')

d_compiled_re = {}
d_compiled_re['email'] = re.compile(r'([0-9a-zA-Z_\.]*\.?\w+@[0-9a-zA-Z._-]*)')

# Either read from full file to create a reduced file (two arguments to readDataFrame)
# Or     read from specified file only.
# In both cases, continue processing

df = readDataFrame("output4.csv", "output_reduced.csv")
#df = readDataFrame("output_reduced.csv")

df = restrictEmailsToYears(df, low=2012, high=2018)
df = addTimeframes(df)
#embed()

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

to = df['To']
cc = df['CC']
sender = df['From']

#----------------------------------------------------------------------
df = cleanDFColumn(df, 'To', d_compiled_re)
df = cleanDFColumn(df, 'CC', d_compiled_re)
df = cleanSenders(df, d_compiled_re)

#-------------------------------
# List of unique senders
from_list = df['From'].values.tolist()
# each element of 'From' is a triplet: first, last, email
unique_senders = list(uniqueEmails(from_list))
unique_senders.sort()


#printList(unique_senders, "unique_senders")
print("nb unique senders with emails: ", len(unique_senders))

#-------------------------------
# List of unique recipients
# Create a list of email triplets
to_list = df['To'].values.tolist()
unique_receivers = set()
for i in range(len(to_list)):
    for lst in to_list[i]:
        unique_receivers.add(lst[2])

unique_receivers = list(unique_receivers)
unique_receivers.sort()

#printSet(unique_receivers, "unique_receivers")
print("nb unique receivers with emails: ", len(unique_receivers))

#-------------------------------
# List of unique recipients
# Create a list of email triplets
cc_list = df['CC'].values.tolist()
unique_cc = set()
for i in range(len(cc_list)):
    for lst in cc_list[i]:
        unique_cc.add(lst[2])

unique_cc = list(unique_cc)
unique_cc.sort()

#printList(unique_cc, "unique_cc")
print("nb unique cc with emails: ", len(unique_cc))
#------------------

cc_triplets = makeListTriplets(df, "CC")
#printList(cc_triplets, "cc_triplets")

to_triplets = makeListTriplets(df, "To")
from_triplets = makeListTriplets(df, "From")
print("nb to_triplets: ", len(to_triplets))
print("nb cc_triplets: ", len(cc_triplets))
print("nb from_triplets: ", len(from_triplets))

full_list = []
full_list.extend(cc_triplets)
full_list.extend(from_triplets)
full_list.extend(to_triplets)
print("nb in full triplet list: ", len(full_list))

# Compute nb of unique elements in full_list based on emails first,
# then based on first, last, email
full_set = set()
for i in full_list:
     full_set.add(tuple(i))  # tuples are hashable, lists are not
print("nb unique triplets: ", len(full_set))

triplets = sortTriplets(list(full_set))
#printList(triplets, "triplets")

#----------------------------------------------------------------------
# triplets: concatenation of From, To, CC, and removal of triplet duplicates.
#    Some elements are missing first and last names, and other elements are missing
#    email.
#
# Create some dictionaries to help process the data
#  d_missing: list of elements from triplets() with missing first and last names.
#      The empty first and last name fields are filled with placeholders in the
#      form of f_x, l_x where x is an integer: 0, 1, 2, ...
#
#  d_triplets: all elements of triplets with missing first and last names replaced by
#  (f_x, l_x). Some elements of d_triplets do not have a first name. Some elements
#  do not have an email.
#
#  d_email: construct from d_triplets, only using elements with first and last names
#    are NOT set to (f_x, l_x).  All emails are real.
#
#  d_missing_items: dictionary constructed from elements in triplets list whose first and
#     last names are (f_x,l_x). They have emails. Then remove the elements that are
#     already present in d_email. Therefore, d_email+d_missing_items contain
#     all the elements. All emails are real.
#
#  new_triplets: constructed
#
#  d_names: complete dictionary by names --> triplet from d_new_triplets
#  d_email: complete dictionary by email --> triplet from d_new_triplets
#
#  Take a triplet from the database, and update it to reflect th reduction in the various fields through uniqueness considerations.
#   1) check triplet email.
#     a) there is an email  ==> use d_email[email]
#     b) there is no email, but there is a name ==> use d_name[name]
#     c) Print out field if a) or b) do not work
# I am not concerned with efficiency. This is a one-time operation.
#
# Once this is done, process columns from database.
#

#----------------------------------------------------------------------
# Consolidate triplets. Fill in missing emails.
d_triplets = {}
d_missing = {}
for i, t in enumerate(triplets):
    if t[0] == '' and t[1] == '':
        arg = ("f_%d"%i, "l_%d"%i)
        d_triplets[arg] = (arg[0], arg[1], t[2])
        d_missing[t[2]] = (arg[0], arg[1], t[2])
    else:
        d_triplets[(t[0],t[1])] = (t[0], t[1], '')
        if t[2] != '':
            d_triplets[(t[0],t[1])] = t

d_triplets_sorted_keys = sortTriplets(list(d_triplets))
#----------------------------------------------------------------------
# Create a new dictionary keyed on the email, only if first and last name are nonzero.
d_email = {}
for k,v in d_triplets.items():
    #print("key: ", k, ", value= ", d_triplets[k])
    if v[0][0:2] != "f_" and v[0][0:2] != "l_":
        d_email[v[2]] = v


#printDict(d_email, "d_email")

print("len(d_email): ", len(d_email))
print("len(d_triplets): ", len(d_triplets))
# number of unique emails in d_triplets

s = set()
for k,v in d_triplets.items():
   s.add(v[2])
print("nb unique emails: ", len(s))
#----------------------------------------------------------------------
# Store the elements of d_triplets that do not have first and last
# names in its own dictionary d_missing. Check email in d_missing  against d_email. If the mail is found, remove the email from d_missing.
# d_missing["email"] = triplet
# the items from d_missing that remain are unique.

print("nb mails in d_missing: ", len(d_missing))
keys_to_remove = []

for k,v in d_missing.items():
    try:
        triplet = d_email[k]
        keys_to_remove.append(k)
    except:
        # email not found in d_email, so keep
        pass

for k in keys_to_remove:
    del d_missing[k]

print("nb mails left in d_missing after filtering: ", len(d_missing))
# This procedure identified about 400 emails

#printDict(d_missing, "d_missing")
#printDict(d_triplets, "d_triplets")

#---------------------------------------------------------------------
#   ALL IS WORKING
#----------------------------------------------------------------------
writeDict("d_triplets_old", d_triplets)
#----------------------------------------------------------------------

# At this stage, there are fields with names and no emails.
# Fill in the missing emails
#new_triplets = []
#print("----------------------------")
#print("New triplets")
#for t in new_triplets:
    #print(t)

for k,v in d_triplets.items():
    #print((t[0], t[1], d_triplets[(t[0], t[1])]))
    print("--> d_triplets, key: ", k, ",  value= ", v)

    #print("d_triplets[('','')], ", d_triplets[('','')])
    #print("t[0], t[1]= ", t[0], t[1])
    n = [k[0], k[1], d_triplets[(k[0], k[1])][2]]
    #print("v= ", v, ",  k= ", k)
    #print("n= ", n)
    if n[2] == '':
        #print("not n2")
        n[2] = "_".join(n[:-1])
    #new_triplets.append(tuple(n))
    d_triplets[k] = tuple(n)
    #print(new_triplets[-1])
    # d_triplets now has all emails

# There are mails with names (f_x, l_x) that exist with real names. These
# emails should be removed.
# create a set with these emails with names (f_x, l_x)
s_emails = set()
for k,v in d_triplets.items():
    try:
        if k[0][1] == '_' and k[1][1] == '_':
            s_emails.add(v[2])
    except:
        pass


# create set A (triplets with f_x, l_x)
# create set B (triplets with f_x, l_x)
d_A = {}
d_B = {}
for k,v in d_triplets.items():
    try:
        if k[0][1] == "_" and k[1][1] == "_":
            d_A[k] = v
        else:
            d_B[k] = v
    except:  # handles one name being ""
        d_B[k] = v

#print("== count= %d ==== "%count)
#print(s_emails)

print("len(d_triplets): ", len(d_triplets))
print("len(d_A): ", len(d_A))
print("len(d_B): ", len(d_B))
print("d_B= ", d_B)

# remove non-unique emails
sB = set()
to_remove = []

for k,v in d_B.items():
    sB.add(v[2])

for kA,vA in d_A.items():
    if vA[2] in sB:
        to_remove.append(kA)

for key in to_remove:
    del d_A[key]

# Maybe not required
sA = set()
for k,v, in d_A.items():
    sA.add(v[2])

print("after duplicate email removals")
print("len(d_A)= ", len(d_A))

# Create d_C,  a set of emails constructed from d_B whose records have a missing first or last name.
# Remove from d_B all records added to d_C
d_C = {}
to_remove = []
print("before d_C: len(d_B)= ", len(d_B))
for k,v in d_B.items():
    if k[0] == '' or k[1] == '':
        d_C[k] = v
        to_remove.append(k)
print("before removal, len(d_C)= ", d_C)

for k in to_remove:
    print("renove from d_C: ", k)
    del d_B[k]

print("after d_C: len(d_B)= ", len(d_B))  # d_B has been reduced
print("after d_C: len(d_C)= ", len(d_C))

# remove from d_C all elements found in d_B
sB = set()
for k,v in d_B.items():
    print("v= ", v)
    sB.add(v[2])
to_remove = []

for kC,vC in d_C.items():
    if vC[2] in sB:
       to_remove.append(kC)

print("before d_C reduction, len(d_C)= ", len(d_C))
for k in to_remove:
    print("to_remove, k= ", k)
    del d_C[k]
print("after d_C reduction, len(d_C)= ", len(d_C))



# https://treyhunner.com/2016/02/how-to-merge-dictionaries-in-python/
#d_final = dict(**d_A, **d_C, **d_B) # only work if **d_C is a string. Here it is a list
#writeDict(d_final, "d_final_unpack")

d_final = d_A.copy()
for k,v in d_C.items():
   d_final[k] = v
for k,v in d_B.items():
   d_final[k] = v


d_email_final = {}
for k,v in d_final.items():
    try:
        d_email_final[v[2]] = v
    except:
        pass

#printDict(d_email_final, "email")
print("len(d_final): ", len(d_final))

writeDict("d_email_final.out", d_email_final)
writeDict("email_dict.out", d_email)
writeDict("d_missing.out", d_missing)
writeDict("d_triplets.out", d_triplets)
writeDict("d_final.out", d_final)
writeDataSeries("from.out", df['From'])
writeDataSeries("to.out", df['To'])
writeDataSeries("cc.out", df['CC'])

#----------------------------------------------------------------------
# Create dictionary of unique emails based on d_final

#----------------------------------------------------------------------
#
import traceback
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------

# list of lists
to_list = processColumn(df, 'To', d_final, d_email_final)
# list of lists
cc_list = processColumn(df, 'CC', d_final, d_email_final)
# list
from_list = processColumn(df, 'From', d_final, d_email_final)

print(len(cc_list))
print(len(to_list))
print(len(from_list))


toPickle(cc_list, "cc_list")
toPickle(to_list, "to_list")
toPickle(from_list, "from_list")
toPickle(d_final, "d_final")

# pickle all the rest columns
l_sent = df['Sent'].tolist()
toPickle(l_sent, 'sent_list')
l_year = df['year'].tolist()
toPickle(l_year, 'year_list')
l_month = df['month'].tolist()
toPickle(l_month, 'month_list')
l_week = df['week'].tolist()
toPickle(l_week, 'week_list')
l_weekday = df['weekday'].tolist()
toPickle(l_weekday, 'weekday_list')
l_day = df['day'].tolist()
toPickle(l_day, 'day_list')
l_dayofyear = df['dayofyear'].tolist()
toPickle(l_dayofyear, 'dayofyear_list')
l_hour = df['hour'].tolist()
toPickle(l_hour, 'hour_list')
# pickle these three lists

#printList(to_list, "processed To")
#printList(cc_list, "processed CC")
#printList(from_list, "processed_from")
#----------------------------------------------------------------------

# save clean dataframe ready for graphing and statistics computation
# This approach is not really useful, since I a dataframe column must
# be a basic type (data is stored in numpy arrays). Strange types are
# converted to strings.
# dataframes with


cols = df.columns
print("df cols= ", cols)
# df.drop(columns=cols[0], inplace=True) # you are dropping the column 'From'
df.to_csv("clean_output_noindex.csv", index=False)
quit()
#----------------------------------------------------------------------
