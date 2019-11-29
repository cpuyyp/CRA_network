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

def readDataFrame(file_name, read_cached=True):
    df1 = pd.read_csv(file_name, index_col=False)

    if not read_cached:  # only set to 1 in order to generate the file output_reduced.csv
        print(df1.columns)
        df = df1.drop(columns=["Subject", "Attachments", "Body"])
        df.to_csv("output_reduced.csv", index_label="Index", index=True)
        cols = df.columns
        df = df.drop(columns=[cols[0], cols[1]]) 
        print(df.head()); quit()

    df = df1
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
re_email = re.compile(r'([0-9a-zA-Z_\.]*\.?\w+@[0-9a-zA-Z._-]*)')
#orig re_email = re.compile(r'.*?([a-zA-Z_]*\.?\w+@[a-zA-Z_]*\.?[a-zA-Z_]*\.?[a-zA-Z]{2,3})')

def extractEmail(string):
    #print("extract string: ", string)
    # Only keep first email found, and remove it from the string
    #print("orig string= ", string)
    print("extract email, string: ", string)
    s = re_email.search(string)

    if s:  # found email string
        print("if s")
        email = re_email.findall(string)[0]
        email = re.sub(r"xa0|\\", " ", email)
        
        # Email is removed from string
        string = string[:s.start()] + string[s.end():]
        # Remove Abbreviations with two or more letters with a dot
        # \s is whitespace
        string = re.sub(r",(\s*[A-Z][\.]?){2,}", " ", string)
        string = re.sub(email, "", string) 
        string = re.sub(r"<>", "", string) 
        string = re.sub(r"\[mailto:\]", "", string) 
        string = re.sub(r"xa0|\\", " ", string)
        string = re.sub(r"\(\)", "", string) 
        print("before Filter 1, string: ", string)
        string = re.sub(r"\b(MD|Major)\b", " ", string)  # MD: Medical Doctor
        print("Filter 1, string: ", string)
        string = string.strip()
        #print("before extract, if: string: ", string)
        string = re.sub(r"[\[<\(]mailto.*?[\]>\)]", " ", string)
        #print("after extract, if: string: ", string)
    else:
        print("else s")
        #  " P.E.   J. Keith Dantin" ==> Keith J. Dantin
        #  Notice the double enclosure. \1 accesses the outer enclosure (( ))
        print("Before, string: ", string)
        string = re.sub(r"(([A-Z][\.]?){2,})\s+([A-Z][\.]?)\s+(\w+)\s+(\w+)", 
                        r"\4 \3 \5", string)
        print("Filter 0, string: ", string)
        #print("removal of abbreviations, string= ", string)
        # PC Wu Asst:  Elaine Mager"" <emager@ci.pensacola.fl.us>,
        # transform to:   Elaine Mager"" <emager@ci.pensacola.fl.us>
        #  ",  P.E.J." ==>  " "
        string = re.sub(r",(\s*[A-Z][\.]?){2,}", " ", string)
        print("Filter 2, string: ", string)
        # Danger: Acceptable is 
        #      first last, abbrev
        #      first middle last, abbrev
        #   Not acceptable
        #      last, first P.E.   (note that there is no comma before the abbreviation in this case)
        email = ''
        print("after extract, else: string: ", string)
    #print("before sub: string= ", string)
    pattern = r"on behalf.*$"
    string = re.sub(pattern, "", string)
    #print("after sub: string= ", string)
    #print("new string= ", string)
    #print("return email: ", email.lower())
    return email.lower(), string


def standardizeName(string): 
# The emails have already been removed. All that is left is a single person's name
    print("standardize string: ", string)
    # transform MikeWood to Mike Wood, for example. 
    string = re.sub(r"\b([A-Z][a-z]+)([A-Z][a-z]+)\b", r"\1 \2", string)
    string = string.strip()
    if len(string) == 0:
        return '', '', ''

    # remove commas and dots
    string = re.sub(r"[<>\(\)\.\"]", "", string)
    # remove non alpha-numeric characters (leave space since it acts as a separator)
    string = re.sub(r"xa0|\\", " ", string)
    #print("standardize, string sub= ", string)

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

    # The csv file sometimes has names in the form "last first middle" instead of "last, first middle"
    # In this case, I must reorder the names. Check the length of middle.strip(). If it is 1, then swap

    first  = first.strip().lower()
    middle = middle.strip().lower()
    last  = last.strip().lower()

    #print("len(last) = ", len(last))  # 504 across all years. 
    #print("first= ", first, ", middle= ", middle, ", last= ", last)
    if len(last) == 1:
        #print("return %s, %s, %s" % (middle, last, first))
        return middle, last, first
    else:
        #print("return %s, %s, %s" % (first, middle, last))
        return first, middle, last



#df = readDataFrame("output4.csv", read_cached=False)
df = readDataFrame("output_reduced.csv", read_cached=True)

df = restrictEmailsToYears(df, low=2012, high=2018)
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
    df.reset_index()  # indices now go from 0, upward
    for irec, rec in enumerate(df[col_name]):
        # string = "gsd BCC: af ;lkjaf] af j]". Must be done before the string is broken up. 
        # Note te use of *?  (non-greedy search). Without it, the search would go to the last bracket
        rec= re.sub(r"(BCC:.*?])", r"]", rec)

        recipient_string = rec[2:-2].replace("'",'')

        if re.search("tosteen@moorebass.com", recipient_string):
            flag = True
        else: 
            flag = False

        # remove strings that have too many words
        if (len(recipient_string.split(' ')) > 1000):
            recipient_string = ''

        r1 = recipient_string.split(';')
        if (len(r1) > 50): 
            #print("number of ; splits: ", len(r1))
            # Do not consider recipient lists greater than 10
            # It is unlikely they are significant
            # Replace recipient list by an empty list
            #df.loc[irec, col_name] = '[""]'
            recipient_string = ''
            r1 = ['']
            

        if len(r1) > 1:
            recipient_list = r1
        elif len(r1) == 1:
           r2 = recipient_string.split(',') 
           # remove strings that have too many fields
           if (len(recipient_string.split(',')) > 50):
               recipient_string = ''
           #if (len(r2) > 10): 
                 #print("number of , splits: ", len(r1))
           if len(r2) == 1:
              recipient_list = r2
              # no further splits necessary. Single name or single email. Pass to standardizeName()
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
        # to be further processed by extractEmail and standardizeName

        for i, person in enumerate(recipient_list):
            email, new_str = extractEmail(person)
            print("triplets: email= ", email, ",  new_str= ", new_str)
            f, m, l = standardizeName(new_str)
            recipient_list[i] = [f,l,email]
            print("triplets: f,l", [f,l])
        recipients.append(recipient_list)
        #if flag: quit()
    print("len(col_name): ", len(df[col_name]))
    print("len(recipients): ", len(recipients))
    df[col_name] = recipients
    return df

#----------------------------------------------------------------------
def cleanSenders(df):
    senders = []
    for rec in df['From']:
        sender = rec[1:-1].replace("'",'')
        email, new_str = extractEmail(sender)
        print("email from sender: ", email)
        #print("new_str= ", new_str)
        f, m, l = standardizeName(new_str)
        print("sender   ",   (f,l))
        # strip() should not be necessary, but the line with "sheila" has an additional leading space
        # I do not know why. 
        senders.append([f,l,email])  # ignore the middle initials
    df["From"] = senders
    return df
#---------------------------------------------------------------
def uniqueEmails(emails):
# emails: list of triplets (first, last, emails)
    emails = np.asarray(emails).copy() # in case

    unique = set()
    for e in emails:
        unique.add(e[2])
    return unique
#-------------------------------
df = cleanDFColumn(df, 'To')
df = cleanDFColumn(df, 'CC')
df = cleanSenders(df)
"""
for row in df['To'].values:
    print("to: ", row)
for row in df['CC'].values:
    print("cc: ", row)
for row in df['From'].values:
    print("from: ", row)
"""

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
#------------------

def makeListTriplets(df, col):
# Make a list of triplets from a database column
# Create a list of email triplets
    triplets = []
    df_list = df[col].values.tolist()
    if col == 'From':
        for row in df_list:
            triplets.append(row)
    else:
        for row in range(len(df_list)):
            for lst in df_list[row]:
                triplets.append(lst)
    return triplets

cc_triplets = makeListTriplets(df, "CC")
for t in cc_triplets:
    print("cc_triplets= ", t)
to_triplets = makeListTriplets(df, "To")
from_triplets = makeListTriplets(df, "From")
#print("triplets")
#print(triplets)
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

def sortTriplets(triplets):
    triplets.sort()
    return(triplets)

triplets = sortTriplets(list(full_set))
for row in triplets:
    print(row)

# Consolidate triplets. Fill in missing emails. 

d_triplets = {}
new_triplets = []
for t in triplets:
    if t[0] == '' and t[1] == '': 
        new_triplets.append(t)
    else:
        d_triplets[(t[0],t[1])] = ''
        if t[2] != '': 
            d_triplets[(t[0],t[1])] = t[2]

triplets = sortTriplets(list(d_triplets))
print("----------------------------")
print("New triplets")
for t in new_triplets:
    print(t)
for t in triplets:
    #print((t[0], t[1], d_triplets[(t[0], t[1])]))
    n = [t[0], t[1], d_triplets[(t[0], t[1])]]
    if not n[2]:
        n[2] = "_".join(n[:-1])
    new_triplets.append(tuple(n))
    print(new_triplets[-1])

#print("+++")
#print("new_triplets: ", new_triplets)
quit()


#------------------

