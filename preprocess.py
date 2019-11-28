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
        email = ''
    #print("before sub: string= ", string)
    pattern = r"on behalf.*$"
    string = re.sub(pattern, "", string.lower())
    #print("after sub: string= ", string)
    #print("new string= ", string)
    return email.lower(), string


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
            recipient_list[i] = [f,l,email]
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
        #print("new_str= ", new_str)
        f, m, l = standardize_name2(new_str)
        #print("   ",   (f,m,l,email))
        # strip() should not be necessary, but the line with "sheila" has an additional leading space
        # I do not know why. 
        senders.append([f,l,email])  # ignore the middle initials
    df["From"] = senders
    return df
#---------------------------------------------------------------
def uniqueEmails(emails):
# emails: list of triplets (first, last, emails)
    #print(emails); quit()
    emails = np.asarray(emails).copy() # in case

    unique = set()
    for e in emails:
        unique.add(e[2])
    return unique
#-------------------------------
df = cleanDFColumn(df, 'To')
df = cleanDFColumn(df, 'CC')
df = cleanSenders(df)

#-------------------------------
# List of unique senders
from_list = df['From'].values.tolist()
# each element of 'From' is a triplet: first, last, email
unique_senders = list(uniqueEmails(from_list))
unique_senders.sort()
for s in unique_senders:
    print("sender: ", s)
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
for s in unique_receivers:
    print("rec: ", s)

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
for s in unique_cc:
    print("cc: ", s)

print("nb unique cc with emails: ", len(unique_cc))

quit()

#--------------------
print("nb_unique_senders= ", nb_unique_senders)
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


