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
        #print(df.head()); quit()

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
    else:  # found no email string
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
    # Remove "jr", "sr" (and variations) from string
    string = re.sub(r"([jJ|sS])[rR]\.?", " ", string)
    # Remove PhD and variations
    string = re.sub(r"[pP]\.?[hH]\.?\s*[dD]\.?", " ", string)
    #print("after sub: string= ", string)
    #print("new string= ", string)
    #print("return email: ", email.lower())
    return email.lower().strip(), string


def standardizeName(string): 
# The emails have already been removed. All that is left is a single person's name
    print("standardize string: ", string)
    # Remove acronyms
    string = re.sub(r"\b([iI]{1,3}|LLC|(RLA|rla))\b", " ", string)
    # A name will never occur before "City of Tallahassee". The email is already extracted.
    string = re.sub(r"\bCity of Tallahassee.*$\b", "", string)
    # Remove single letter words and acronyms
    #string = re.sub(r"\b([A-Za-z]\.?|[iI]{1,3}|LLC|(RLA|rla))\b", " ", string)
    # Remove words that are clearly not names, and might appear with the names
    string = re.sub(r"\b([Ee]xecutive|[Dd]irector)\b", " ", string)
    # Remove items in parenthesis (mail is already extracted) (non-greedy)
    string = re.sub(r"\(.*?\)", '', string)
    # Remove single letter words [F. or F] and acronyms (will this work here?)
    string = re.sub(r"\b([A-Za-z0-9]\.?|[iI]{1,3}|(LLC|llc)|(RLA|rla))\b", " ", string)
    # Remove multiple capital words. Dangerous because one can have JT as a first name.
    
    # transform MikeWood to Mike Wood, for example. 
    string = re.sub(r"\b([A-Z][a-z]+)([A-Z][a-z]+)\b", r"\1 \2", string)
    # fix previous transformation. Mc, De, Van, Von, get reattached
    string = re.sub(r"\b(Mc|De|Van|Von) ([A-Z][a-z]+)\b", r"\1\2", string)
    # Some string end in a comma with spaces as a result of preprocessing. Remove comma
    # Sometimes, there is a dangly double quote or backslash at the end
    string = re.sub(r",\s*[\"\\-]?\s*$", "", string)
    print("after last comma removed, string= ", string)
    # remove any words with an "@" sign, whether email or not. Sometimes emails linger
    #string = re.sub(r"([\w\._-]+@[\w\._-])+", r"", string)
    #print("standardize, after filter, string: ", string

    string = string.strip()
    if len(string) == 0:
        return '', '', ''

    # remove dots, <>, (),  and &. Leave commas
    string = re.sub(r"[<>\(\)\.\"&]+", "", string)
    # remove non alpha-numeric characters (leave space since it acts as a separator)
    string = re.sub(r"xa0|\\", " ", string)
    # remove non-letters, followed by space at the end of the string. just to be sure. 
    print("before last filter, string= ", string)
    string = re.sub(r"[\W]+\s*$", "", string)
    print("standardize, string sub= ", string)


    strings = string.split(',')
    first = last = middle = ''

    print("process string: ", string)

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
        print("return %s, %s, %s" % (middle, last, first))
        print("Should not happen. I removed single letter initials.")
        quit()
        return middle, last, first
    else:
        #print("return %s, %s, %s" % (first, middle, last))
        return first, middle, last.split(" ")[-1]



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
            print("\n")
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
        print("\n")
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
#for s in unique_receivers:
    #print("rec: ", s)

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
#for s in unique_cc:
    #print("cc: ", s)

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
#for t in cc_triplets:
    #print("cc_triplets= ", t)

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
for k in d_triplets_sorted_keys:
    print("triplets= ", d_triplets[k])
#print("d_triplets")
#print(d_triplets)
#----------------------------------------------------------------------
# Create a new dictionary keyed on the email, only if first and last name are nonzero.
d_email = {}
for k,v in d_triplets.items(): 
    #print("key: ", k, ", value= ", d_triplets[k])
    if v[0][0:2] != "f_" and v[0][0:2] != "l_":
        d_email[v[2]] = v 


for i, (k,v) in enumerate(d_triplets.items()):
    print("%d, d_triplets, k,v= "%i, v)

for i, (k,v) in enumerate(d_email.items()):
    print("%d, d_email, k,v= "%i, v)

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

for k,v in d_missing.items():
    print("d_missing: ", v)

# all fields are filled. Use l_x, f_x if there were empties. 
for k,v in d_triplets.items():
    print("d_triplets: ", v)
#---------------------------------------------------------------------
#   ALL IS WORKING
#----------------------------------------------------------------------
def writeDict(file_name, dic):
    fd = open(file_name, "w")
    for k,v in dic.items():
        print(k, v, file=fd)
    fd.close()

def writeDataSeries(file_name, ds):
    fd = open(file_name, "w")
    for d in ds:
        print(d, file=fd)
    fd.close()

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

writeDict("email_dict.out", d_email)
writeDict("d_missing.out", d_missing)
writeDict("d_triplets.out", d_triplets)
writeDataSeries("from.out", df['From'])
writeDataSeries("to.out", df['To'])
writeDataSeries("cc.out", df['CC'])
#----------------------------------------------------------------------
# 
def processTriplet(i, triplet, d_missing, d_triplets):
    # find first and last
    if triplet[2]:
        try:
            triplet = d_missing[triplet[2]]
        except:
            pass
    # find email
    else:
        try:
            email = d_triplets[tuple(triplet[0:2])][2] 
            triplet[2] = email
        except:
            pass
    pass
    print("%d, processTriplet, return triplet= "%i, triplet)
    return triplet

#----------------------------------------------------------------------

print("cc_triplets")
for i,triplet in enumerate(cc_triplets):
    processTriplet(i, triplet, d_missing, d_triplets)

print("to_triplets")
for i,triplet in enumerate(to_triplets):
    processTriplet(i, triplet, d_missing, d_triplets)

print("from_triplets")
for i,triplet in enumerate(from_triplets):
    processTriplet(i, triplet, d_missing, d_triplets)

#writeDataSeries("afrom.out", df['From'])  # a for after
#writeDataSeries("ato.out", df['To'])
#writeDataSeries("acc.out", df['CC'])

print("End") 
quit()
quit()
#----------------------------------------------------------------------

for t in cc_triplets():
   print("cc: ", t)
quit()


#print("+++")
#print("new_triplets: ", new_triplets)


#------------------
# I know have a clean list of triplets
# 
#  cc_triplets
#  from_triplets
#  to_triplets
# 
# I need to replace the contents of the (cc,from,to) dataframe columns
# by the cleaned triplets. 
#
# Given a triplet (t0,t1,t2) in the database, replace it by 
#   (t0,t1,d_triplets[(t[0], t[1])])
#

print(df['From'].head())
#a  = df['From'].apply(lambda x: (x[0],x[1],d_triplets[(x[0],x[1])]))

for t in df['From'].values:
    print(d_email[(t[2])])


print(a)
