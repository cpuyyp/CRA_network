
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import re
import pickle
from IPython import embed
from collections import defaultdict

import traceback
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def readDataFrame(in_filename, cached_filename=None):
# if cached_file_name is not None, write a cached file (smaller in size)
# if cached_file_name is None: read in_file_name and use the associated he dataframe
    df = pd.read_csv(in_filename, usecols=['From', 'Sent', 'To', 'CC'])
    cols = df.columns
    print("df columns: ", cols)

    if cached_filename:  # only set to 1 in order to generate the file cached_filename
        df.to_csv(cached_filename, index_label="Index", index=False)
        cols = df.columns
        print("Cached data in file: ", cached_filename)
        df = pd.read_csv(cached_filename, index_col=False)

    df['Sent']    = pd.to_datetime(df['Sent'])
    df = df.reset_index(drop=True)
    return df


def restrictEmailsToYears(df, low=2013, high=2014):
    # restrict emails
    df = df[df['Sent'] > datetime( low, 1, 1, 0, 0, 0)]
    df = df[df['Sent'] < datetime(high, 1, 1, 0, 0, 0)]
    return df

def extractEmail(string, compiled_re):
    s = compiled_re['email'].search(string)

    if s:  # found email string
        #print("if s")
        email = compiled_re['email'].findall(string)[0]
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
        string = re.sub(r"\b(MD|Major)\b", " ", string)  # MD: Medical Doctor
        string = string.strip()
        string = re.sub(r"[\[<\(]mailto.*?[\]>\)]", " ", string)
    else:  # found no email string
        #  " P.E.   J. Keith Dantin" ==> Keith J. Dantin
        #  Notice the double enclosure. \1 accesses the outer enclosure (( ))
        string = re.sub(r"(([A-Z][\.]?){2,})\s+([A-Z][\.]?)\s+(\w+)\s+(\w+)",
                        r"\4 \3 \5", string)
        # PC Wu Asst:  Elaine Mager"" <emager@ci.pensacola.fl.us>,
        # transform to:   Elaine Mager"" <emager@ci.pensacola.fl.us>
        #  ",  P.E.J." ==>  " "
        string = re.sub(r",(\s*[A-Z][\.]?){2,}", " ", string)
        # Danger: Acceptable is
        #      first last, abbrev
        #      first middle last, abbrev
        #   Not acceptable
        #      last, first P.E.   (note that there is no comma before the abbreviation in this case)
        email = ''
    pattern = r"on behalf.*$"
    string = re.sub(pattern, "", string)
    # Remove "jr", "sr" (and variations) from string
    string = re.sub(r"([jJ|sS])[rR]\.?", " ", string)
    # Remove PhD and variations
    string = re.sub(r"[pP]\.?[hH]\.?\s*[dD]\.?", " ", string)
    return email.lower().strip(), string


def standardizeName(string):
# The emails have already been removed. All that is left is a single person's name
    #print("standardize string: ", string)
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
    #print("after last comma removed, string= ", string)
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
    #print("before last filter, string= ", string)
    string = re.sub(r"[\W]+\s*$", "", string)
    #print("standardize, string sub= ", string)


    strings = string.split(',')
    first = last = middle = ''

    #print("process string: ", string)

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

    # The csv file sometimes has names in the form "last first middle" instead of "last, first middle"
    # In this case, I must reorder the names. Check the length of middle.strip(). If it is 1, then swap

    first  = first.strip().lower()
    middle = middle.strip().lower()
    last  = last.strip().lower()

    if len(last) == 1:
        print("return %s, %s, %s" % (middle, last, first))
        print("Should not happen. I removed single letter initials.")
        quit()
        return middle, last, first
    else:
        #print("return %s, %s, %s" % (first, middle, last))
        return first, middle, last.split(" ")[-1]
#----------------------------------------------------------------------
def cleanDFColumn(df, col_name, d_compiled_re):
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
                   #print("-----------------------------------------")
                   #print("recipient_string= ", recipient_string)
                   #pattern = r"(\b\w+\b),(\b\w_\b)"
                   pattern = r"\s*\w+\s*,\s*\w+\s*"
                   m = re.search(pattern, recipient_string)
                   if not m:
                       recipient_list = recipient_string.split(',')
                       #print("recipient_list= ", recipient_list)
                   else:
                       nb_commas = len(re.findall(",", recipient_string))
                       #print("nb commas= ", nb_commas)
                       if nb_commas > 2:  # not foolproof. There might be only two emails.
                           recipient_list = recipient_string.split(',')
                       else:
                           recipient_list = [recipient_string]
                       # Separate by commas
        #-------------------

        # At this stage, recipient_list is the list of recipients
        # to be further processed by extractEmail and standardizeName

        for i, person in enumerate(recipient_list):
            #print("\n")
            email, new_str = extractEmail(person, d_compiled_re)
            #print("triplets: email= ", email, ",  new_str= ", new_str)
            f, m, l = standardizeName(new_str)
            recipient_list[i] = [f,l,email]
            #print("triplets: f,l", [f,l])
        recipients.append(recipient_list)
    #print("len(col_name): ", len(df[col_name]))
    #print("len(recipients): ", len(recipients))
    df[col_name] = recipients
    return df
#----------------------------------------------------------------------
def cleanSenders(df, d_compiled_re):
    senders = []
    for rec in df['From']:
        sender = rec[1:-1].replace("'",'')
        #print("\n")
        email, new_str = extractEmail(sender, d_compiled_re)
        #print("email from sender: ", email)
        #print("new_str= ", new_str)
        f, m, l = standardizeName(new_str)
        #print("sender   ",   (f,l))
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
def printList(the_list, name):
    for element in the_list:
       print("%s: " % name, element)
#-------------------------------
def printSet(the_set, name):
    for element in the_set:
       print("%s: " % name, element)
#----------------------------------------------------------------------
def printDict(the_dict, name):
    for k,v in the_dict.items():
       print("%s: " % name, k, v)
#-------------------------------
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
#-------------------------------
def sortTriplets(triplets):
    triplets.sort()
    return(triplets)
#---------------------------------------------------------------------
def writeDict(file_name, dic):
    fd = open(file_name, "w")
    for k,v in dic.items():
        print(k, v, file=fd)
    fd.close()
#----------------------------------------------------------------------
def writeList(file_name, lst):
    fd = open(file_name, "w")
    for k in lst:
        print(k, file=fd)
    fd.close()
#---------------------------------------------------------------------
def writeDataSeries(file_name, ds):
    fd = open(file_name, "w")
    for i,d in enumerate(ds):
        print(i, d, file=fd)
    fd.close()

#---------------------------------------------------------------------
def processTriplet(ix, triplet, d_final, d_email_final):
    # d_A: dictionary:
    # find first and last
    # triplet can have missing components
    # 1. missing names: search email in s
    print("enter triplet: ", triplet)

    try:
        triplet = d_email_final[triplet[2]]
        #print("completed try: triplet= ", triplet)
    except:
        #print("enter except")
        try:
            #print("   enter try: triplet= ", triplet)
            triplet = d_final[tuple(triplet[0:2])]
            #print("   completed  try: triplet= ", triplet)
        except Exception as error:
            #print("enter final except: triplet= ", triplet)
            #logger.exception(error)
            cond = not (triplet[0] == "" and triplet[1] == "" and triplet[2] == "")
            if cond:
                # the only expected outcome here is ("", "", "")
                print("processTriplet: should not happen (", triplet, ")")

    print("return triplet: ", triplet)
    return triplet

#----------------------------------------------------------------------
def processColumn(df, col, d_final, d_email_final):
# Go through the column, creating a list of triplets.
# Make a list of triplets from a database column
# Create a list of email triplets
# Each row is a triplet

    # Each row is a single triplet
    if col == 'From':
        triplets = []
        df_list = df[col].values.tolist()
        for j, row in enumerate(df_list):
            triplet = processTriplet(j, row, d_final, d_email_final)
            triplets.append(triplet)
        return triplets

    # Each row is a list of triplets
    else:
        triplet_list_list = []
        df_list = df[col].tolist()
        #for row in range(len(df_list)):
        for i, df_row in enumerate(df_list):
            #print("=== row %d" % i)
            triplet_list = []
            for j,lst in enumerate(df_row):
                #print("processTriplet, lst[%d]= "%j, lst)
                triplet = processTriplet(j, lst, d_final, d_email_final)
                triplet_list.append(triplet)
            triplet_list_list.append(triplet_list)
        return triplet_list_list

#----------------------------------------------------------------------


#----------------------------------------------------------------------

def dictFreq(in_dict):
# generate frequency distribution of dictionary values

    d_freq = defaultdict(int)  # initialize to zero by default
    for k,v in in_dict:
        d_freq[v] += 1

    for k,v in d_freq:
        print(v)

#----------------------------------------------------------------------
def listFreq(in_list):
# generate frequency distribution of dictionary values

    d_freq = defaultdict(int)  # initialize to zero by default
    for k in in_list:
        d_freq[k] += 1

    for k,v in d_freq:
        print(v)

#----------------------------------------------------------------------
def toPickle(a_list, name):
    with open(name+".pickle", 'wb') as handle:
        pickle.dump(a_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------
def fromPickle(name):
    with open(name+".pickle", 'rb') as handle:
        return  pickle.load(handle)

#----------------------------------------------------------------------
def nameToIndexDict(l_unique_names):
# create name to index dictionary and index to name dictionary for later use
# returns (name2id, id2name)
    name2id = {}
    for idx,name in enumerate(l_unique_names):    # unique_names
        name2id[name] = idx

    id2name = {}
    for idx, name in enumerate(l_unique_names):
        id2name[idx] = name

    return name2id, id2name

#----------------------------------------------------------------------
def sendReceiveList(l_from, l_to, l_cc, max_rec, max_to):
# create a list of send-rec pairs. Ignore messages that have more
# than max_rec recipients and more than max_cc cc: mails.

    d_pairs = defaultdict(list)
    send = l_from   # list of triplets
    to   = l_to     # list of lists of triplets
    cc   = l_cc     # list of list of triplets

    #for i in range(10):
        #print("to[%d]= "%i, len(to[i]))

    for i,s in enumerate(send):
        if (len(to[i]) > max_to):
            #print("> max, len(t)= ", len(to[i]))
            continue
        #print("==> keep, to[%d]= "%i, to[i]);
        for j,t in enumerate(to[i]):
            #if to[0] == "": continue
            #print("to: ", t, ",   s= ", s)
            d_pairs[s[2]].append(t[2])

    return d_pairs

#----------------------------------------------------------------------
def createConnectionMatrix(unique_names, name2id, l_from, l_to, l_cc):
    s_to_r = np.zeros((len(unique_names),len(unique_names)), dtype='int')
    for i, s in enumerate(l_from):
        try:
            sx = name2id[s]   # x means integer index
        except:
            continue
        for r in l_cc[i]:
            if r != ['', '', ''] and r != ('', '', ''):
                try:
                    rx = name2id[r]
                    s_to_r[sx, rx] += 1
                except:
                    pass
        for t in l_to[i]:
            try:
                tx = name2id[t]
                s_to_r[sx, tx] += 1
            except:
                pass
    return s_to_r

#----------------------------------------------------------------------
def addTimeframes(df):
    df['year']    = df['Sent'].dt.year
    df['month']   = df['Sent'].dt.month
    df['week']    = df['Sent'].dt.week
    df['weekday'] = df['Sent'].dt.weekday
    df['day'] = df['Sent'].dt.day
    df['dayofyear'] = df['Sent'].dt.dayofyear
    df['hour']    = df['Sent'].dt.hour
    return df
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
