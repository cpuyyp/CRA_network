import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import re
from function_library import *

#--------------------------------------------------------
# Search for "On Behalf" in the string.
re_behalf = re.compile(r'(.*)[Oo]n [Bb]ehalf')

# Search anywhere on the line for <mailto:xxxx> where xxx is the mailing address. 
# The patterns also consider () and [] instead of <>. Extract the last occurence of 
# an mail address via ().
re_bracket = re.compile(r'(.*)[\[\<\(]mailto\:(.*?)[\]\>\)]')

# Search anywhere on the line for <xxxx>, (xxxx), or [xxxx]. Extract xxxx. If there are multiple 
# occurences, extract the last one because .* is greedy.
re_bracket2 = re.compile(r'(.*)[\[\<\(](.*?)[\]\>\)]')

# Fix email retrieved by bracket: removed special ending and beginning non-alphanumer characters, and 
# remove any spaces

# Search anywhere in the line for a sequence of lower and upper cases letters, followed by a space (\s)
# followed by 0 or 1 capital letters [A-Z]? followed by 0 or a dots \.? For example, "GORdon A."
# ERROR: only allowing for a single space after the first name. Should be one or more spaces (\s+ intead of \s?) ?
# After the (first_name + initial + dot), allow for one sapce, and a last name formed from 1 or more [A-Za-z]. 
# Extract the last name with ()
#re_name1 = re.compile(r'.*?([A-Za-z]+\s?[A-Z]?\.?)\s([A-Za-z]+)')  # Written by Joey

# I would allow for multiple initials: Gordon A. Z. Erlebacher, or Gordon A Z Erlebacher (dots might be missing), 
#  or Gordon AZ Erlebacher  (Multiple names, or concatenated initials)
# re_name1 has no comma
# PROBLEM: if the first name has unicode characters, it is not found, and the first name   switches to the last name. 
# That is probably the result of \b, and the fact I allow zero or more sections to the first name. 
# I would like to capture the first name REGARDLESS OF THE CHARACTERS IN THE NAME, even numbers. So use \w and the UNICODE FLAG.
re_name1 = re.compile(r""".*?   
    (                         # Capture first name +  middle names (abbreviated or not)
        #[A-Za-z]+             # First name (one or more characters)
        (?: \s?
            #(?:\b[A-Za-z]+?\b\.?\s*)+   # Not captured: Abbreviation structure
            (?:\b\w+?\b\.?\s*)+   # Not captured: Abbreviation structure
        )+                       # Not captured: 1j or more sequence of abbreviations
        #)?                       # Not captured: 0 or more sequence of abbreviations
    )                            # end of first name + initials capture
    #\s*\b([A-Za-z]+\b)             # Capture last name preceded by one or more spaces. 
    \s*\b(\w+\b)             # Capture last name preceded by one or more spaces. 
                                 # \b forces the last name to start at a word boundary
""", re.X)

# Extract last name with (), followed by a comma, (the last name only contains lower and upper case, no apostrophes or dashes)
# Leave a single space, followed by first name ([A-Za-z], zero or more (?) spaces (\s) (ERROR: should be 1 ore more spaces (+), 
# followed by 0 or 1 initials [A-Z] followed by 0 or 1 dots (\.?). NOT GENERAL ENOUGH. What about Erle, Gordon A.E.F. or 
#   Erle, Gordon A. E.   F. ? 
#re_name2 = re.compile(r'.*?([A-Za-z]+),\s([A-Za-z]+\s?[A-Z]?\.?)')  # Joey's version
# Gordon's version
re_name2 = re.compile(r""".*?
    #([A-Za-z]+)\b,\s*                  # Capture last name followed by comma
    (\w+)\b,\s*                  # Capture last name followed by comma
    (                                  # Capture first name and initials
        (?: \s*
            (?:\b\w+\b\.?\s*)?   # Not captured: Abbreviation structure
        )*                             # Not captured: 0 or more sequence of abbreviations
    )   
""", re.X)  

re_name3 = re.compile(r""".*?
    ([\w^0-9]+)
""", re.X | re.UNICODE)



# Capture an email: that is a very complex exercise, so it it unlikely that this approarch works, but it is likely good enough. 
# Here is an more complex solution: https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s01.html
# Email string is defined as the first occurance of the expression in (): 
#   The email is a series of [a-zA-Z_] followed by zero or one dots \.? followed by one or more letter/number (/w), followed by '@'. 
#   followed by [a-zA-Z_0-9] (equiv to [\w] zero or more times, followed by a dot (0 or 1 times), followed by [a-zA-Z_] 0 or more, 
#  followed by period 0 or more, followed by 2 or three letters 
re_email =  re.compile(r'.*?([a-zA-Z_]*\.?\w+@[a-zA-Z_0-9]*\.?[a-zA-Z_]*\.?[a-zA-Z]{2,3})')

#  The following special characters are allowed in an email name:  ! # $ % & ' * + - / = ? ^ _ ` { |
#  For now, we ignore them. 
#  A domain suffix is required, so the domain after @ has at least one period. The full domain must be less than 64 characters long. 
#  We ignore this constraint. 
# Hyphens are allowed, but must be surrounded by characters: (?:[A-Za-z0-9]+\-?)+[A-Za-z0-9]+
# Domain name rules: https://www.dynadot.com/community/blog/domain-name-rules.html
#                    https://www.20i.com/support/domain-names/domain-name-restrictions
re_email1 = re.compile(r'.*?([\w.]*@[A-Za-z0-9\-]*\.?[a-zA-Z_]*\.?[a-zA-Z])')  # not used

re_domain = re.compile(r""".*?(  
     (?: (?:  [A-Za-z0-9]+\-?)+[A-Za-z0-9]+\.)+ (?: [A-Za-z]+)
)""", re.X)

# Rewritten by G. Erlebacher, 2022-02-13.
re_email = re.compile(r"""(.*?)(
     [\w.]*@    # email name: upper/lower case, numerals, dots, underscores
     (?:        # non-captured domain name
         (?:    # non-captured
             [A-Za-z0-9]+\-?    # sequence of letters/numbers followed by one hyphen (`seqA`)
         )+                     # one or more of `seqA`
         [A-Za-z0-9]+\.         # one or more letters after the last hyphen, followed by a dot
     )+                         # non-capture: one or more of `seqB`
     (?: [A-Za-z]+)             # the final domain segment, after the last dot
    )   # capture full email
""", re.X)

re_first_last = re.compile(r'^([A-Z][a-z]*)([A-Z][a-z]*)$')






#--------------------------------------------------------

def check_name(tname, tname_orig, temail_orig, unrecognized_names):
    if re_name2.match(tname): 
        name = re_name2.findall(tname)[0]
        first_name = name[1]
        last_name = name[0]
        # if first_name == '' or last_name == '':
        #     print("--> full name: ", tname)
        #     print("     first: ", first_name, "    last: ", last_name)
    elif re_name1.match(tname):
        name = re_name1.findall(tname)[0]
        first_name = name[0]
        last_name = name[1]
        # if first_name == '' or last_name == '':
        #     print("==> full name: ", tname)
        #     print("     first: ", first_name, "    last: ", last_name)
    else:
        first_name = ''   # I am throwing away the incorrect string
        last_name = 'unrecognized'
        # How many letters are capitalized in tname_orig
        # search = re.search(r'(?:[A-Z][^A-Z]+){2}', tname_orig)
        # if len(re_cap.findall(tname_orig)) == 2:
        first_last = re_first_last.match(tname_orig.strip())
        if first_last: 
            first_name = first_last.group(1).lower()
            last_name = first_last.group(2).lower()
            # print("*** valid name: ", tname_orig, first_name, last_name)
        else:
            unrecognized_names.add((tname_orig.strip(), temail_orig.strip()))
        
    return first_name.strip(), last_name.strip()
#----------------------------------------------------------

def check_email(f, regex):
    tname, temail = regex.findall(f)[0]
    tname=tname.lower().strip()
    temail=temail.lower()
    return tname, temail
#----------------------------------------------------------

def check_email1(f, unrecognized_names):
    bracket = re_bracket.match(f)
    bracket2 = re_bracket2.match(f)
    email = re_email.match(f)
    # print("f: ", f)
    if bracket:
        # print("bracket ")
        temail = bracket.groups()[1]
        tname = bracket.groups()[0]
    elif bracket2:
        # print("bracket 2")
        temail = bracket2.groups()[1]
        tname = bracket2.groups()[0]
    elif email:
        # print("re_email")
        temail = email.groups()[1]
        tname = email.groups()[0]
    else:
        temail = ""
        tname = f

    temail = re.sub("[\s\?]+", "", temail)
    first_name, last_name = check_name(tname, tname, temail, unrecognized_names)
    # print("tname: ", tname, ",  temail: ", temail)
    # print("           first: ", first_name, ", last: ", last_name)
    return first_name.strip(), last_name.strip(), temail.strip()
#----------------------------------------------------------

def ge_search_from(from_list, email_to_names, name_to_emails, unrecognized_names):
    for f in from_list:
        if pd.isnull(f):
            # print("from is null: ", f)
            continue
        is_behalf = ''
        if re_behalf.match(f):
            f = re_behalf.findall(f)[0]
            is_behalf = 'b_'

        first, last, email = check_email1(f, unrecognized_names)
        email_to_names[email].add((is_behalf+first, is_behalf+last))
        name_to_emails[(first, last)].add(is_behalf+email)
#----------------------------------------------------------

def ge_search_list_of_lists(the_list, email_to_names, name_to_emails, unrecognized_names):
    # print("==> search_to_section")
    for ts in the_list: 
        if pd.isnull(ts):  # if nan
            continue
        # ts = ts.lower()
        ts = ts.split(';')
        for t in ts:
            t = t.strip("'")
            first, last, email = check_email1(t, unrecognized_names)
            email_to_names[email].add((first, last))
            name_to_emails[(first, last)].add(email) 
#----------------------------------------------------------

def overlap(name, email):
    # return the number of common characters
    
    email = set(re.sub("@.*$","", email))
    first = re.sub("^_b", "", name[0])
    last  = re.sub("^_b", "", name[1])
    new_name = set(re.sub("[^\w]", "", first+last))
    # print("email: ", email, ",  name: ", name)
    return len(email.intersection(new_name))
#----------------------------------------------------------

def compute_email_to_chosen_name(email_to_names):
    # An email should have only one name associated with it. 

    null_chosen_names = []
    email_to_chosen = {}
    email_to_names_with_periods = {}

    for email, names in email_to_names.items():
        # Remove lines that are likely incorrect
        if len(email) > 50 or len(names) > 5: continue
        # Do not consider an email with no "@"
        if not re.match(".*@", email): continue
        if email == "": continue
        lg = 0
        chosen_name = ''
        for name in names:
            if name[1] == 'unrecognized': continue
            # if first or last name contains a dot, flag it: 
            if re.match(r'.*\.[a-z]', name[0]) or re.match(r'.*\.[a-z]', name[1]):
                email_to_names_with_periods[email] = name
                continue
            if re.match(r'\w*(:?gov|_?com|us|edu)$', name[1]):
                if not re.match(r'.*\wus', name[1]):
                    email_to_names_with_periods[email] = name
                    continue
            lgo = overlap(name, email)
            if lgo > lg:
                lg = lgo
                chosen_name = name
        if chosen_name == '':
            null_chosen_names.append((email, names))
        else:
            email_to_chosen[email] = chosen_name

    return null_chosen_names, email_to_chosen, email_to_names_with_periods

#--------------------------------------------------------

def compute_name_to_chosen_email(name_to_emails):
    # An email should have only one name associated with it.

    null_chosen_emails = []
    names_to_remove = [] # not used
    name_to_chosen = {}

    for name, emails in name_to_emails.items():
        # Remove lines that are likely incorrect
        if len(emails) > 10 or len(name) > 50: continue
        chosen_email = ''
        lg = 0
        for email in emails:
            lgo = overlap(name, email)
            if lgo > lg:
                lg = lgo
                chosen_email = email
        if chosen_email == '':
            null_chosen_emails.append((name, emails))
        else:
            name_to_chosen[name] = chosen_email

    return null_chosen_emails, names_to_remove, name_to_chosen
#--------------------------------------------------------

def round_trip_name(name_to_chosen, email_to_chosen):
    emails_not_found_in_email_to_chosen = []
    non_name_match = []
    name_match = []

    # Is name_to_chosen and email_to_chosen consistent? Let us find out? 
    for name, email in name_to_chosen.items():
        if len(email) > 50: continue
        try:
            new_name = email_to_chosen[email]
        except:
            # print("Exception, name: ", name, "email: ", email)
            emails_not_found_in_email_to_chosen.append(email)
            continue

        if name != new_name:
            non_name_match.append((name, new_name))
        else:
            name_match.append((name, new_name))
            # print("name: ", name, ",   new_name: ", new_name)

    print("non name match: ", len(non_name_match))
    print("name match: ", len(name_match))
    print("emails_not_found_in_email_to_chosen: ", len(emails_not_found_in_email_to_chosen))
    return emails_not_found_in_email_to_chosen, non_name_match, name_match
#--------------------------------------------------------

def round_trip_email(email_to_chosen, name_to_chosen):
    names_not_found_in_name_to_chosen = []
    non_email_match = []
    email_match = []

    # Is name_to_chosen and email_to_chosen consistent? Let us find out? 
    for email, name in email_to_chosen.items():
        if len(name) > 50: continue
        try:
            new_email = name_to_chosen[name]
        except:
            # print("Exception, name: ", name, "email: ", email)
            names_not_found_in_name_to_chosen.append(name)
            continue

        if email != new_email:
            non_email_match.append((email, new_email, name))
        else:
            email_match.append((email, new_email, name))
            # print("name: ", name, ",   new_name: ", new_name)

    non_email_match.sort(key=lambda x: x[0])
    email_match.sort(key=lambda x: x[0])
    names_not_found_in_name_to_chosen.sort(key=lambda x: x[0])

    print("non email match: ", len(non_email_match))
    print("email match: ", len(email_match))
    print("names_not_found_in_name_to_chosen: ", len(names_not_found_in_name_to_chosen))
    return names_not_found_in_name_to_chosen, non_email_match, email_match
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
