import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import regex as rex
from function_library import *
from copy import copy
from unidecode import unidecode
import name_matching_lib as nmlib

rex.DEFAULT_VERSION = rex.VERSION1

#--------------------------------------------------------
# Search for "On Behalf" in the string.
re_behalf = rex.compile(r'(.*)[Oo]n [Bb]ehalf')

# Search anywhere on the line for <mailto:xxxx> where xxx is the mailing address. 
# The patterns also consider () and [] instead of <>. Extract the last occurence of 
# an mail address via ().
re_bracket = rex.compile(r'(.*)[\[\<\(]mailto\:(.*?)[\]\>\)]')

# Search anywhere on the line for <xxxx>, (xxxx), or [xxxx]. Extract xxxx. If there are multiple 
# occurences, extract the last one because .* is greedy.
re_bracket2 = rex.compile(r'(.*)[\[\<\(](.*?)[\]\>\)]')

# multiple internal words with two or more cap letters, sandwiched between two names starting with caps
re_name_caps_name = rex.compile(r'\s?([A-Z][a-z]+)\s+(?:[A-Z]{2,}\s)+?([A-Z][a-z]+)\Z')

# Fix email retrieved by bracket: removed special ending and beginning non-alphanumer characters, and 
# remove any spaces

# Search anywhere in the line for a sequence of lower and upper cases letters, followed by a space (\s)
# followed by 0 or 1 capital letters [A-Z]? followed by 0 or a dots \.? For example, "GORdon A."
# ERROR: only allowing for a single space after the first name. Should be one or more spaces (\s+ intead of \s?) ?
# After the (first_name + initial + dot), allow for one sapce, and a last name formed from 1 or more [A-Za-z]. 
# Extract the last name with ()
#re_name1 = rex.compile(r'.*?([A-Za-z]+\s?[A-Z]?\.?)\s([A-Za-z]+)')  # Written by Joey

# I would allow for multiple initials: Gordon A. Z. Erlebacher, or Gordon A Z Erlebacher (dots might be missing), 
#  or Gordon AZ Erlebacher  (Multiple names, or concatenated initials)
# re_name1 has no comma
# PROBLEM: if the first name has unicode characters, it is not found, and the first name   switches to the last name. 
# That is probably the result of \b, and the fact I allow zero or more sections to the first name. 
# I would like to capture the first name REGARDLESS OF THE CHARACTERS IN THE NAME, even numbers. So use \w and the UNICODE FLAG.
re_name1 = rex.compile(r""".*?   
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
""", rex.X | rex.UNICODE)

# Extract last name with (), followed by a comma, (the last name only contains lower and upper case, no apostrophes or dashes)
# Leave a single space, followed by first name ([A-Za-z], zero or more (?) spaces (\s) (ERROR: should be 1 ore more spaces (+), 
# followed by 0 or 1 initials [A-Z] followed by 0 or 1 dots (\.?). NOT GENERAL ENOUGH. What about Erle, Gordon A.E.F. or 
#   Erle, Gordon A. E.   F. ? 
#re_name2 = rex.compile(r'.*?([A-Za-z]+),\s([A-Za-z]+\s?[A-Z]?\.?)')  # Joey's version
# Gordon's version
re_name2 = rex.compile(r""".*?
    #([A-Za-z]+)\b,\s*                  # Capture last name followed by comma
    (\w+)\b,\s*                  # Capture last name followed by comma
    (                                  # Capture first name and initials
        (?: \s*
            (?:\b\w+\b\.?\s*)?   # Not captured: Abbreviation structure
        )*                             # Not captured: 0 or more sequence of abbreviations
    )   
""", (rex.X | rex.UNICODE)  )

re_name3 = rex.compile(r""".*?
    ([\w^0-9]+)
""", rex.X | rex.UNICODE)

# Remove (some punctuation, phd, jr, mr, ms, mrs, dr, iii, ii
re_honorifics = rex.compile(r'\b([\.,_-]|p\.?h\.?d\.?|j\.?r\.?|mr\.|ms\.|mrs\.|dr\.|ii|iii)', rex.I)

# Capture an email: that is a very complex exercise, so it it unlikely that this approarch works, but it is likely good enough. 
# Here is an more complex solution: https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s01.html
# Email string is defined as the first occurance of the expression in (): 
#   The email is a series of [a-zA-Z_] followed by zero or one dots \.? followed by one or more letter/number (/w), followed by '@'. 
#   followed by [a-zA-Z_0-9] (equiv to [\w] zero or more times, followed by a dot (0 or 1 times), followed by [a-zA-Z_] 0 or more, 
#  followed by period 0 or more, followed by 2 or three letters 
re_email =  rex.compile(r'.*?([a-zA-Z_]*\.?\w+@[a-zA-Z_0-9]*\.?[a-zA-Z_]*\.?[a-zA-Z]{2,3})')

#  The following special characters are allowed in an email name:  ! # $ % & ' * + - / = ? ^ _ ` { |
#  For now, we ignore them. 
#  A domain suffix is required, so the domain after @ has at least one period. The full domain must be less than 64 characters long. 
#  We ignore this constraint. 
# Hyphens are allowed, but must be surrounded by characters: (?:[A-Za-z0-9]+\-?)+[A-Za-z0-9]+
# Domain name rules: https://www.dynadot.com/community/blog/domain-name-rules.html
#                    https://www.20i.com/support/domain-names/domain-name-restrictions
re_email1 = rex.compile(r'.*?([\w.]*@[A-Za-z0-9\-]*\.?[a-zA-Z_]*\.?[a-zA-Z])')  # not used

re_domain = rex.compile(r""".*?(  
     (?: (?:  [A-Za-z0-9]+\-?)+[A-Za-z0-9]+\.)+ (?: [A-Za-z]+)
)""", rex.X)

# Rewritten by G. Erlebacher, 2022-02-13.
# Simplified version of allowable emails
re_email = rex.compile(r"""(.*?)(
     [\w\-.]*@    # email name: upper/lower case, numerals, dots, underscores
     (?:        # non-captured domain name
         (?:    # non-captured
             [A-Za-z0-9]+\-?    # sequence of letters/numbers followed by one hyphen (`seqA`)
         )+                     # one or more of `seqA`
         [A-Za-z0-9]+\.         # one or more letters after the last hyphen, followed by a dot
     )+                         # non-capture: one or more of `seqB`
     (?: [A-Za-z]+)             # the final domain segment, after the last dot
    )   # capture full email
""", rex.X)

re_first_last = rex.compile(r'^([A-Z][a-z]*)([A-Z][a-z]*)$', rex.UNICODE)
re_remove_internal_initials = rex.compile(r'\A\s*(\w+\s+)(?:\w+\.?\s?)*(\s\w+)\Z', rex.UNICODE)

#re_remove_front_initials = rex.compile(r'\A(?>\w+\.?\s?)*(\w+\s+\w+)\Z', rex.UNICODE)
re_remove_front_initials = rex.compile(r"""\A
        (?>\w+\.?\s?)*     # Atomic: once pattern is found, do not retrace.
                           #  multi-letters & dot(s) & space(s) (the entire structure repeated)
        (\w+\s+\w+)        #  word & space & word
        \Z""", rex.UNICODE | rex.X)

#re_domain = rex.compile(r'.*\. (?:com|edu|gov|us|org|[\w\.]+\.\s+\w+)\Z') #, flags=rex.I)
re_domain = rex.compile(r'.*[a-z]\.\s+(?:\w+\Z)') #, flags=rex.I)
#  Remove all names ending in a dot followed by a space, and then \w+ (capture these to see them)
re_remove_suffixes = rex.compile(r'\s+(?:III)\Z', flags=rex.I)
#re_breakup_camelcase = rex.compile(r'\A

def explode_camelcase(name):
    # explode camelcase '\1 \2' except for certain words like VanGordon. 
    return rex.sub(r'(?=\w+$)([a-z])(?<!(?:Cur|La|Ha|Mac|Mc|Di|De|Van|Du))([A-Z])', r'\1 \2', name, flags=rex.UNICODE)
    #return rex.sub(r'([a-z])([A-Z])', r'\1 \2', name)

def remove_suffixes(name): 
    return rex.sub(' (?:III)\Z', '', name, rex.I)

def remove_internal_caps(name):
    match = re_name_caps_name.match(name)
    if match:
        groups = match.groups()
        return groups[0] + ' ' + groups[1]
    else:
        return name

def remove_internal_initials(name):
    name1 = re_remove_internal_initials.findall(name)
    if name1 == []:
        return name
    else:
        return "".join(name1[0])

def remove_front_initials(name):
    #print("remove_front: name: ", name)
    name1 = re_remove_front_initials.findall(name)
    if name1 == []:
        return name
    else:
        return "".join(name1[0])

def remove_multi_spaces(name):
    return rex.sub(r'\s+', ' ', name)

def remove_non_words(name):
    return rex.sub(r'[\.\-_,;!?\'\"]', '', name)

def join_prefix_to_stem(name):
    return  rex.sub(r'\A(Cur|La|Ha|Mac|Mc|Di|De|Van|Du) (\w+)\Z', r'\1\2', name, flags=rex.U | rex.I)

def remove_honorifics(name): 
    #name = rex.sub(r'(?<!\w)\s+(?:jr|dr|mr|mrs|ms|phd|md|tdia)', '', name, flags=rex.I | rex.U)
    name = rex.sub(r'\b(?:jr|sr|dr|mr|mrs|ms|phd|md|tdia)\b', '', name, flags=rex.I | rex.U)
    name = rex.sub(r'\b(?:Rep|AE|AIA|AICP)\b', '', name, rex.U)
    # Some of the initials could be middle names in disguise. So perhaps these should be removed
    # after other processing is done
    return name
    #return rex.sub(r'\A(dr|mr|mrs|ms|ph\.d\.)\.*\s?', '', name, flags=rex.I | rex.U)


def starts_with_number(name):
    return rex.sub(r'\A\d.*\Z', 'unrecognized', name)

def remove_domains(name): 
    if re_domain.match(name):
    # Also remove names that end with a dot: "gordon. com"
        return "unrecognized"
    else:
        return name
    return 

VERBOSE = False

def clean_name(name):
    """
    Standardize name
    Parameters
    ----------
    name : string
        The string of characters to normalized

    Return
    ------
        return the cleaned name
    """
    if VERBOSE: print("--- init name: ", name)
    name = unidecode(name)
    name = remove_multi_spaces(name)
    name = remove_honorifics(name)
    if VERBOSE: print("  after remove_honorifics: ", name)
    name = remove_domains(name)
    if VERBOSE: print("  after remove_domains: ", name)
    name = rex.sub(' (III\Z)', '', name, rex.I)  
    if VERBOSE: print("  after remove_suffixes: ", name)
    name = remove_internal_initials(name)   #  REMOVE THIS
    if VERBOSE: print("  after remove_domains: ", name)
    name = remove_front_initials(name)  # REMOVE THIS
    if VERBOSE: print("  after remove_front: ", name)
    name = remove_non_words(name)
    if VERBOSE: print("  after remove_non_w: ", name)
    name = starts_with_number(name)   # PERHAPS SIMPLY REMOVE THE NUMBER
    #name = explode_camelcase(name)
    name = remove_multi_spaces(name)
    if VERBOSE: print("  after remove_multis_paces: ", name)
    #name = remove_internal_caps(name)
    return name.lower()
    name = rex.sub(r"\s*(Deputy County Administrator|Swift Creek Middle School Assistant Principal|Executive\s?Director|Neighborhood\s?Services|(iNKBRIDGE)?\s+Business Fax)", "", name, flags=rex.I)
    name = rex.sub(r"\s*(Human\s?Resources|solid\s?waste|commissioner|orthodontics|community board|CHIARAPODERI)", "", name, flags=rex.I)
    name = rex.sub(r"representative", "tallahassee", name, flags=rex.I)
    name = rex.sub(r'construction business fax', "", name, flags=rex.I)
    name = rex.sub(r'public works', "", name, flags=rex.I)
    name = rex.sub(r'labor analytics', "", name, flags=rex.I)
    name = rex.sub(r'assistant', "", name, flags=rex.I)
    name = rex.sub(r'Electronet Broadband Communications  Business Fax', "", name, flags=rex.I)
    name = rex.sub(r'Commission on the Status of Women and Girls', "", name, flags=rex.I)
    # if this case is not removed, the string is probably too long and is removed from consideration
    # Automation would be nice. While this string contains a single email, some strings are long and do not. 
    # or some strings contain a date/time, or a cost ($420), which is obviously in error. 
    if rex.match(r'.*Division', name) and rex.match(r'.*Historical', name):
        print("1. MATCH, name: ", name)
        name = rex.sub(r'Division of Historical Resources Department', '', name, flags=rex.I)
        print("2. MATCH, name: ", name)
    name = rex.sub(r'All Saints District Community Association', '', name, flags=rex.I)
    name = rex.sub(r'Downtown Tallahassee Business Associatio', '', name, flags=rex.I)
    name = rex.sub(r'Federal and Foundation Assistance Monitor', '', name, flags=rex.I)
    # Remove On behalf of and everything before

def remove_job_descriptors(name):
    name = rex.sub(r"\s*(Deputy County Administrator|Swift Creek Middle School Assistant Principal|Executive\s?Director|Neighborhood\s?Services|(iNKBRIDGE)?\s+Business Fax)", "", name, flags=rex.I)
    name = rex.sub(r"\s*(Human\s?Resources|solid\s?waste|commissioner|orthodontics|community board|CHIARAPODERI)", "", name, flags=rex.I)
    name = rex.sub(r"representative", "tallahassee", name, flags=rex.I)
    name = rex.sub(r'construction business fax', "", name, flags=rex.I)
    name = rex.sub(r'public works', "", name, flags=rex.I)
    name = rex.sub(r'labor analytics', "", name, flags=rex.I)
    name = rex.sub(r'assistant', "", name, flags=rex.I)
    name = rex.sub(r'Electronet Broadband Communications  Business Fax', "", name, flags=rex.I)
    name = rex.sub(r'Commission on the Status of Women and Girls', "", name, flags=rex.I)
    # if this case is not removed, the string is probably too long and is removed from consideration
    # Automation would be nice. While this string contains a single email, some strings are long and do not. 
    # or some strings contain a date/time, or a cost ($420), which is obviously in error. 
    if rex.match(r'.*Division', name) and rex.match(r'.*Historical', name):
        print("1. MATCH, name: ", name)
        #name = rex.sub(r'Division of Historical Resources Department', '', name, flags=rex.I)
        name = rex.sub(r'Florida Department, Division of Historical Resources, Grants Department', '', name, flags=rex.I)
        print("2. MATCH, name: ", name)
    name = rex.sub(r'All Saints District Community Association', '', name, flags=rex.I)
    name = rex.sub(r'Downtown Tallahassee Business Associatio', '', name, flags=rex.I)
    name = rex.sub(r'Federal and Foundation Assistance Monitor', '', name, flags=rex.I)
    return name

def clean_name_better(name):
    """
    Standardize name
    Parameters
    ----------
    name : string
        The string of characters to normalized

    Return
    ------
        return the cleaned name
    """
    name = unidecode(name)
    VERBOSE = False
    if rex.match(r'.*Florida', name) and rex.match(r'.*Historical', name):
        print("VERBOSE True")
        VERBOSE = True
    # check_name is where "last, first" is transformed to "first last"
    # However, there was no check for multiple commas. The switch should only be made if there is only a single comma. 
    name = rex.sub(r'\A([^,]+),([^,]+)\Z', r'\2 \1', name)  # Only a single comma
    if VERBOSE: print("--- init name: ", name)
    name = remove_multi_spaces(name)
    name = remove_honorifics(name)
    #if VERBOSE: print("  after remove_honorifics: ", name)
    name = remove_domains(name)
    #if VERBOSE: print("  after remove_domains: ", name)
    name = rex.sub(' (III\Z)', '', name, rex.I)  
    #if VERBOSE: print("  after remove_suffixes: ", name)
    #name = remove_internal_initials(name)   #  REMOVE THIS
    #if VERBOSE: print("  after remove_domains: ", name)
    #name = remove_front_initials(name)  # REMOVE THIS
    #if VERBOSE: print("  after remove_front: ", name)

    name = remove_job_descriptors(name)

    # Remove punctuation
    name = remove_non_words(name)
    #if VERBOSE: print("  after remove_non_w: ", name)
    #name = starts_with_number(name)   # PERHAPS SIMPLY REMOVE THE NUMBER
    #name = explode_camelcase(name)
    # Remove numbers separated by spaces
    name = re.sub(r'\b\d+\b', '', name)
    # Handle army names
    if rex.match(r'.*(USARMY)', name):
        # Should I remove spaces from the name? 
        name = rex.sub(r'(NG FLARNG|NGFLARNG|NGFLARNG|FLARNG)', '', name)
        name = rex.sub(r'USARMY', '', name)
        name = rex.sub(r' US ', ' ', name)
        name = rex.sub(r'(SFC|MSG|SGT|1ST|SSG|COL|SFC|III|WMAJ|SPC|1SG|MAJ|EDSON|CPT|II|SGM|JrLTC|LTC|CSM|NG FLANG|1LT|NGFLANG|NG|NDARNG|NFG|CIVNG)', '', name)
    if rex.match(r'.*FLARNG', name):
        name = rex.sub('NG\s*FLARNG.*US', '', name)
        name = rex.sub(r'(SFC|MSG|SGT|1ST|SSG|COL|SFC|III|WMAJ|SPC|1SG|MAJ|EDSON|CPT|II|SGM|JrLTC|LTC|CSM|NG FLANG|1LT|NGFLANG|NG|NDARNG|NFG|CIVNG|CTR|CIV)', '', name)
    if rex.match(r'.*NG\s+FLANG\s+US', name):
        name = rex.sub('NG\s+FLANG\s+US', '', name)
        name = rex.sub(r'(SFC|MSG|SGT|1ST|SSG|COL|SFC|III|WMAJ|SPC|1SG|MAJ|EDSON|CPT|II|SGM|JrLTC|LTC|CSM|NG FLANG|1LT|NGFLANG|NG|NDARNG|NFG|CIVNG|CTR|CIV)', '', name)

    # Remove more honorifics. Not clear whether I should since I lose information about the person. 
    # Perhaps the honorific should stay together with the name. 
    # NOTE: Processing Jr. can be tricky. For example: "Gordon Erlebacher, Jr. Joey Jingze" in the To: field probably refers to two names  with 
    # Jr. attached to the first name. 

    #if rex.match(r'.*behalf', name, flags=rex.I):
    #    print(name)
    if rex.match(r'.*on behalf of', name, flags=rex.I):
        #print("bef: ", name)
        name = rex.sub(r'^.*on behal. of', '', name, flags=rex.I)
        #print("aft: ", name)

    #  '5Gor' => 'Gor'
    name = re.sub(r'\d+([A-Z])', r'\1', name)
    #if VERBOSE: print("  after remove digits: ", name)
    name = remove_multi_spaces(name)
    #if VERBOSE: print("  after remove_multi_spaces: ", name)
    name = remove_internal_caps(name)
    if rex.match(r'ARMY', name): # DEBUG
        print("ARMY MATCH. SHOULD NOT HAPPEN")

    VERBOSE = False
    return name.lower()

#--------------------------------------------------------

def check_name(tname, tname_orig, temail_orig, unrecognized_names):
    if re_name2.match(tname): 
        name = re_name2.findall(tname)[0]
        first_name = name[1]
        last_name = name[0]
    elif re_name1.match(tname):
        name = re_name1.findall(tname)[0]
        first_name = name[0]
        last_name = name[1]
    else:
        first_name = ''   # I am throwing away the incorrect string
        last_name = 'unrecognized'
        first_last = re_first_last.match(tname_orig.strip())
        if first_last: 
            first_name = first_last.group(1).lower()
            last_name = first_last.group(2).lower()
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

def check_email1(f, unrecognized_names, field_dict):
    bracket = re_bracket.match(f)
    bracket2 = re_bracket2.match(f)
    email = re_email.match(f)
    if bracket:
        temail = bracket.groups()[1]
        tname = bracket.groups()[0]
    elif bracket2:
        temail = bracket2.groups()[1]
        tname = bracket2.groups()[0]
    elif email:
        temail = email.groups()[1]
        tname = email.groups()[0]
    else:
        temail = ""
        tname = f

    temail = rex.sub("[\s\?]+", "", temail)
    first_name, last_name = check_name(tname, tname, temail, unrecognized_names)
    first_name = first_name.strip()
    last_name = last_name.strip()
    full_name = first_name + ' ' + last_name
    temail = temail.strip()
    field_dict[f] = (full_name, temail)
    return first_name, last_name, temail
    #return full_name, temail
#-----------------------------------------------------------
def check_name_first(tname, unrecognized_names):
    if re_name2.match(tname): 
        name = re_name2.findall(tname)[0]
        first_name = name[1]
        last_name = name[0]
    elif re_name1.match(tname):
        name = re_name1.findall(tname)[0]
        first_name = name[0]
        last_name = name[1]
    else:
        first_name = ''   # I am throwing away the incorrect string
        last_name = 'unrecognized'
        first_last = re_first_last.match(tname.strip())
        if first_last: 
            first_name = first_last.group(1)
            last_name = first_last.group(2)
        else:
            unrecognized_names.add(tname)
        
    return first_name.strip() + ' ' + last_name.strip()
#----------------------------------------------------------
def extract_name_email(f, unrecognized_names):
    bracket = re_bracket.match(f)
    bracket2 = re_bracket2.match(f)
    email = re_email.match(f)
    if bracket:
        temail = bracket.groups()[1]
        tname = bracket.groups()[0]
    elif bracket2:
        temail = bracket2.groups()[1]
        tname = bracket2.groups()[0]
    elif email:
        temail = email.groups()[1]
        tname = email.groups()[0]
    else:
        temail = ""
        tname = f

    temail = rex.sub("[\s\?]+", "", temail)
    full_name = check_name_first(tname, unrecognized_names)
    temail = temail.strip()
    return full_name, temail
#----------------------------------------------------------

def ge_search_from(from_list, email_to_names, name_to_emails, unrecognized_names, field_dict):
    for f in from_list:
        # Dictionary values will be updated at a later time
        if pd.isnull(f):
            continue
        is_behalf = ''
        if re_behalf.match(f):
            f = re_behalf.findall(f)[0]
            is_behalf = ''  # Do not modify the name

        first, last, email = check_email1(f, unrecognized_names, field_dict)
        email_to_names[email].add((is_behalf+first, is_behalf+last))
        name_to_emails[(first, last)].add(is_behalf+email)
#----------------------------------------------------------

def ge_search_list_of_lists(the_list, email_to_names, name_to_emails, unrecognized_names, field_dict):
    # print("==> search_to_section")
    for ts in the_list: 
        if pd.isnull(ts):  # if nan
            continue
        # ts = ts.lower()
        ts = ts.split(';')
        for t in ts:
            t = t.strip("'")
            first, last, email = check_email1(t, unrecognized_names, field_dict)
            email_to_names[email].add((first, last))
            name_to_emails[(first, last)].add(email) 
#----------------------------------------------------------
# create dictionary of all elements from To:, From:, Cc: to itself: el -> el
def create_field_dict(from_list, to_list, cc_list):
    field_dict = defaultdict(str)
    for f in from_list:
        if pd.isnull(f):
            continue
        field_dict[f] = f

    for el in cc_list:
        if pd.isnull(el):
            continue
        ts = el.split(';')    
        ## Should sometimes split by comma, which leads to difficulties
        ## Erlebacher, Gordon, Zhang, Joey (cannot tell breaks)
        ## Gordon Erlebacher, Joey Zhang  (can tell breaks. So if all elements have at least two words)
        ## If number of commas == number of emails - 1, then break by commas is safe. NOT DONE. 
        for f in ts:
            field_dict[f] = f

    for el in to_list:
        if pd.isnull(el):
            continue
        ts = el.split(';')
        for f in ts:
            field_dict[f] = f

    return field_dict
#----------------------------------------------------------
def clean_field_dict_values(field_dict, nb_el_to_process= 10000000, remove_if_longer_than=40):
    # Process field_dict: simplify all names
    # field_dict values already contained simplifed names. Here I am repeating n
    # the work with slightly different routines
    # return field_dict1: dict str => tuple(clean name, email) (unicodes replaced)
    unique_names = defaultdict(set)
    removed = []
    unrecognized_names = set()
    field_dict1 = defaultdict(tuple)
    #field_list =list(field_dict.items())

    for i, (k,v) in enumerate(field_dict.items()):
        name = v  # a single string
        name0 = copy(name)
        if i > nb_el_to_process: break

        # Best to call this prior to extracting name and email
        name = remove_job_descriptors(name)
        n, e = extract_name_email(name, unrecognized_names)

        name = n
        temail = e

        flag = False

        #if rex.match(r'.*Florida', k) and rex.match(r'.*Historical', k):
        if rex.match(r'.*GAY RIGHTS BACKERS GET BOOST IN CONSERVATIVE COMMUNITY', k):
            print("field_dict: ", field_dict[k])
            print("=> MATCH, key: ", k)
            print("   MATCH, name: ", name)
            print("   MATCH, temail: ", temail)
            print("   MATCH, name1: ", name1)
            flag = True

        #if i > 540 and i < 550: print(f"({i}) {name}")
        name = rex.sub('\"', '', name)
        name1 = clean_name_better(name)

        """
        if rex.match(r".*Florida Professional Firefighters  fpf@fpf.org", k):
            print("matched: Florida Professional Firefighters  fpf@fpf.org")
            print("v: ", v)
            print(f"n: {n},   e: {e}")
            print("name1: ", name1)
            print("len(name): ", len(name1))
        """

        if len(name1) > remove_if_longer_than: 
            #print(f"REMOVE, {len(name)}: name")
            field_dict1[name0] = 'invalid'
            removed.append(name)
            continue

        # Remove any email
        #name1 = rex.sub(r'[\"\'\s]+(\[mailto:|<mailto:|<\w+@|\A\w+@).*\Z', '', name1, rex.I | rex.UNICODE)
        name1 = rex.sub(r'[\"\'\s]+(\[mailto:|<mailto:|<\w+@|\A\w+@).*\Z', '', name1, rex.I | rex.UNICODE)
        if rex.match(r'@', name1):
            print("MATCH @ (SHOULD NOT OCCUR): ", name1)
        # 'unrecognized': neither re_name1 or re_name2 match
        if name1 != 'unrecognized': 
            name2 = explode_camelcase(name1)   # no effect since everything is lowercase
            name1 = clean_name_better(name1).strip()   # Why is this one needed (there are no more capitalizations)
            # key: cleaned name
            # value: set of From, To, CC fiels
            # List of unique names
            unique_names[name1].add(v)
        #field_dict1[name0] = (name1, temail.lower())
        #if rex.match(r".*Florida Professional Firefighters  fpf@fpf.org", name0):
            #print("matched: Florida Professional Firefighters  fpf@fpf.org")

        # Remove spaces from names
        field_dict1[name0] = (rex.sub(r'\s', '', name1), temail.lower())
        #if flag == True:
            #print("True, name0: ", name0)
        #print("--------------------------")
    return field_dict1, unique_names, removed, unrecognized_names
#----------------------------------------------------------
def overlap_str_str(str1, str2):
    str1 = set(str1)
    str2 = set(str2)
    return len(str1.intersection(str2)), len(str1), len(str2)
#----------------------------------------------------------
# I would like to compute continous overlap. That means using some form of regex
def overlap_tuple_str(name, email):
    # return the number of common characters
    
    email = set(rex.sub("@.*$","", email))
    first = rex.sub("^_b", "", name[0])
    last  = rex.sub("^_b", "", name[1])
    new_name = set(rex.sub("[^\w]", "", first+last))
    #print("email: ", email, ",  name: ", new_name)
    return len(email.intersection(new_name))
#----------------------------------------------------------
def str_to_set_lower(str_to_set):
    # return new dictionary with elements in value set lowered
    # the keys remain unchanged
    # Will only work if the values are lists of strings. More general versions could lower
    # the keys, and check whether the keys are a tuple.
    str_to_set_lower = defaultdict(set)
    for i, (k,v) in enumerate(str_to_set.items()):
        v1 = set()
        for el in v:
            if el != "":
                v1.add(el.lower())
        str_to_set_lower[k] = v1
    return str_to_set_lower
#----------------------------------------------------------

def compute_email_to_chosen_name(email_to_names):
    """
    An email should have only one name associated with it. 
    Return a sequence of 3 arrays
        - null_chosen_names: chosen name is ''
        - email_to_chosen
        - email_to_names_with_periods
    """

    null_chosen_names = []
    email_to_chosen = {}
    email_to_names_with_periods = {}

    for email, names in email_to_names.items():
        #print("email=> ", email)
        # Remove lines that are likely incorrect
        if len(email) > 50 or len(names) > 5: continue
        # Do not consider an email with no "@"
        if not rex.match(".*@", email): continue
        if email == "": continue
        lg = 0
        chosen_name = ''
        lst = []
        lgo_lst = []
        compare_what = []
        for name in names:
            lst.append(name)
            #print("names= ", names)
            if name[1] == 'unrecognized': continue
            # if first or last name contains a dot, flag it: 
            if rex.match(r'.*\.[a-z]', name[0]) or rex.match(r'.*\.[a-z]', name[1]):
                email_to_names_with_periods[email] = name
                continue
            if rex.match(r'\w*(:?gov|_?com|us|edu)$', name[1]):
                if not rex.match(r'.*\wus', name[1]):
                    email_to_names_with_periods[email] = name
                    continue
            # Added lower()
            name1 = name.lower() #(name[0].lower(), name[1].lower())
            #lgo = overlap_tuple_str(name1, email.lower())
            lgo, len1, len2 = overlap_str_str(name1, rex.sub("@.*$", "", email.lower()))
            compare_what.append((name1, email.lower()))
            lgo_lst.append(lgo)
            if lgo > lg:
                lg = lgo
                chosen_name = name
        if chosen_name == '':
            null_chosen_names.append((email, names))
            print("null name list: ", lst, "    email ", email, "   lgo_lst: ", lgo_lst)
            print("     compare_what: ", compare_what)
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
            # Lower case comparison
            #lgo = overlap(name, email)
            name1 = (name[0].lower(), name[1].lower())
            lgo = overlap(name1, email.lower())
            if lgo > lg:
                lg = lgo
                chosen_email = email
        if chosen_email == '':
            null_chosen_emails.append((name, emails))
        else:
            name_to_chosen[name] = chosen_email

    return null_chosen_emails, names_to_remove, name_to_chosen
#--------------------------------------------------------

def tuple_lower(name):
    return (name[0].lower(), name[1].lower())
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


        #if name != new_name:
        if tuple_lower(name) != tuple_lower(new_name):
            non_name_match.append((name, new_name, email))
        else:
            name_match.append((name, new_name, email))
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

# Original functions from Joey Jingze Zhang

# find unique complete person in From section
def search_from_section(from_list):
    print("==> search_from_section")
    for f in from_list[0:200]:
        if pd.isnull(f):
            continue
        f = f.lower()
        if re_behalf.match(f):
            f = re_behalf.findall(f)[0]

        first, last, email = check_email1(f)
        if email == "":
            print(f"email'', first: {first}, last: {last}, email: {email}")
        if first == "":
            print(f"first'', first: {first}, last: {last}, email: {email}")
        if last == "":
            print(f"last'', first: {first}, last: {last}, email: {email}")

        if email not in named_email_list:
            person = (first, last, email)
            people_list.append(person)
            named_email_list.append(email)
            name_list.append(first + ' ' + last)  # should not be needed

# find unique complete person in TO section
def search_to_section(to_list):
    print("==> search_to_section")
    for ts in to_list[0:200]:
        if pd.isnull(ts):  # if nan
            continue
        ts = ts.lower()
        ts = ts.split(';')
        for t in ts:
            t = t.strip("'")
            email_match = re_email.match(t)
            bracket_match = re_bracket.match(t)
            bracket2_match = re_bracket2.match(t)
            if bracket_match:
                print("bracket: ", bracket_match.groups())
            if bracket2_match:
                print("bracket2: ", bracket2_match.groups())
            continue

            if re_bracket.match(t) or re_bracket2.match(t):
                if re_bracket.match(t):
                    tname, temail = re_bracket.findall(t)[0]
                else:
                    tname, temail = re_bracket2.findall(t)[0]
                tname_orig = tname
                temail_orig = temail
                tname=tname.lower().strip()
                temail=temail.lower()
                if tname == '':
                    continue

                if re_email.match(temail) and re_email.match(tname) == None:
                    email = re_email.findall(temail)[0]
                    if len(tname.split()) != 2:
                        first_name = tname
                        last_name = ' '
                    else:
                        first_name, last_name = check_name(tname, tname_orig, temail_orig)

                    if email not in named_email_list:
                        person = (first_name, last_name, email)
                        people_list.append(person)
                        named_email_list.append(email)
                        name_list.append(first_name + ' ' + last_name)

def search_cc_section(cc_list):
    print("==> search_cc_section")
    # find unique complete person in CC section
    for ccs in cc_list:
        if pd.isnull(ccs):
            continue
        #ccs = ccs.lower()  # not needed. tname is lowered further down
        ccs = ccs.split(';')
        for cc in ccs:
            cc = cc.strip("'")
            if re_bracket.match(cc) or re_bracket2.match(cc):
                if re_bracket.match(cc):
                    tname, temail = re_bracket.findall(cc)[0]
                else:
                    tname, temail = re_bracket2.findall(cc)[0]
                tname_orig = tname
                tname=tname.lower().strip()
                temail_orig = temail
                temail=temail.lower()
                if tname == '':
                    continue

                if re_email.match(temail) and re_email.match(tname) == None:
                    email = re_email.findall(temail)[0]
                    if len(tname.split()) != 2:
                        first_name, last_name = check_name(tname, tname_orig, temail_orig)
                    else:
                        first_name, last_name = check_name(tname, tname_orig, temail)

                    if email not in named_email_list:
                        person = (first_name, last_name, email)
                        people_list.append(person)
                        ## Obviously, person must have only two commas)
                        split_person0 = person[0].split(',')
                        if len(split_person0) != 1:  # SHOULD NOT HAPPEN in a perfect world
                            print("SHOULD NOT HAPPEN: split_person: ", split_person0)  # <<< Identifies errors
                            print("person: ", person)
                            print("   tname: ", tname)
                            print()
                        named_email_list.append(email)
                        name_list.append(first_name + ' ' + last_name)
#--------------------------------------------------------

def analyze_from_list(from_list):
    new_from_list = []

    unknown_idx = 0
    # replace the From section with unique people information
    # from_list: all the names from the From: column, without removing duplicates
    # Purpose: ...

    for f in from_list[0:200]:
        if pd.isnull(f):  # NaN
            person = ('f'+str(unknown_idx), 'l'+str(unknown_idx), 'f'+str(unknown_idx)+'_'+'l'+str(unknown_idx))
            unknown_idx = unknown_idx + 1
            new_from_list.append(person)
            continue
        email_exist_flag = 0
        f = f.lower().strip("'")  #  "Why single quote?
        email = ''
        first_name = ''
        last_name = ''
        if re_behalf.match(f):
            # f should be first/last name + email
            f = re_behalf.findall(f)[0]
            # print("re_behalf, f= ", f)


        email_match = re_email.match(f)
        if email_match:
            # print("re_email matched, f: ", f)
            # print("findall: ", re_email.findall(f))
            email = email_match.groups()[-1]
            first_last = email_match.groups()[0:-1]
            # print("email_match, email: ", email)
            # print("       f: ", f)
            # print("             first_last: ", first_last)
            if re_name2.match(first_last[0]):
                # print("         re_name2 match: ", re_name2.match(f).groups())
                last_name, first_name = re_name2.match(first_last[0]).groups()[0:2]
                # print("name2: ", first_name, last_name)
            elif re_name1.match(first_last[0]):
                # print("         re_name1 match: ", re_name1.match(f).groups())
                # print("first_last: ", first_last)
                first_name, last_name = re_name1.match(first_last[0]).groups()[0:2]
                # print("name1: ", first_name, last_name)
            else:
                # print("NO NAME MATCH, t: ", f)
                first_name = 'fake'
                last_name = 'fake'

            email = email.lower()
            email_exist_flag = 1
            ## If email exists, why not add it to the named_email_list?
            ## ANSWER: because named_email_list requires first/last name + email
        else:
            # print("No email match, f= ", f)
            if re_name2.match(f):
                # print("         re_name2 match: ", re_name2.match(f).groups())
                last_name, first_name = re_name2.match(f).groups()[0:2]
                # print("name2: ", first_name, last_name)
            elif re_name1.match(f):
                # print("         re_name1 match: ", re_name1.match(f).groups())
                first_name, last_name = re_name1.match(f).groups()[0:2]
                # print("name1: ", first_name, last_name)
            else:
                # if there is no name match, invent first and last names
                # print("NO NAME MATCH, f: ", f)
                # first_name = 'fake'
                # last_name = 'fake'
                first_name = 'f'+str(unknown_idx)
                last_name = 'l'+str(unknown_idx)
                unknown_idx = unknown_idx + 1
        # if len(f.split()) != 2:    # CHECK
        #     first_name = f
        #     last_name = ' '     # why space and not empty
        # else:   # two words separated by space
        #     if re_name1.match(f):
        #         name = re_name1.findall(f)[0]
        #         first_name = name[0]
        #         last_name = name[1]
        #     elif re_name2.match(f):
        #         name = re_name2.findall(f)[0]
        #         first_name = name[1]
        #         last_name = name[0]
        #     elif email_exist_flag == 1:
        #         pass
        #     else:
        #         print('error: cannot find name and email in f')
        #         print('f:', f)
        #         first_name = 'fake'
        #         last_name = f

        name = first_name + ' ' + last_name
        if first_name == 'fake':
            print(f"(fake name), f: {f}\n          first: {first_name}, last: {last_name}")
            pass

        ### NEED NEW DATASTRUCTURES

        # all entries in named_email_list have a valid entry in people_list at the same index
        if email in named_email_list:   # named_email_list: only emails
            idx = named_email_list.index(email)
            new_from_list.append(people_list[idx])
        elif name in name_list:   ## WE MIGHT NOT NEED THIS (GE). Handle person with multiple emails
            idx = name_list.index(name)
            new_from_list.append(people_list[idx])
        elif email_exist_flag == 1:   # email by itself
            person = ('f'+str(unknown_idx), 'l'+str(unknown_idx), email)
            print(f"==> mail by itself, f: {f}, \n          person: ", person)
            print("              name: ", name)
            new_from_list.append(person)
            unknown_idx = unknown_idx + 1
        else:
            person = (first_name, last_name, first_name + '_' + last_name)
            new_from_list.append(person)
#--------------------------------------------------------

def analyze_list_of_lists(list_of_lists):
    new_to_lists = []

    # replace the To section with unique people information

    for ts in to_list:
        if pd.isnull(ts):
            new_to_lists.append([])
            continue
        ts = ts.lower()
        ts = ts.split(';')
        new_to_list = []
        for t in ts:       # ts: list of recipients
            t = t.strip("'")
            email_exist_flag = 0
            email = ''
            first_name = ''
            last_name = ''
            if re_behalf.match(t):
                t = re_behalf.findall(t)[0]

            if re_email.match(t):
                email = re_email.findall(t)[0]
                email = email.lower()
                email_exist_flag = 1
            if len(t.split()) != 2:
                first_name = t
                last_name = ' '
            else:
                if re_name1.match(t):
                    name = re_name1.findall(t)[0]
                    first_name = name[0]
                    last_name = name[1]
                elif re_name2.match(t):
                    name = re_name2.findall(t)[0]
                    first_name = name[1]
                    last_name = name[0]
                elif email_exist_flag == 1:
                    pass
                else:
                    first_name = 'fake'
                    last_name = t
                    print('error: cannot find name and email in t, make fake name')
                    print(f"t: {t},  first: {first_name}, last: {last_name}")
    #                 break

            name = first_name + ' ' + last_name


            if email in named_email_list:
                idx = named_email_list.index(email)
                new_to_list.append(people_list[idx])
            elif name in name_list:
                idx = name_list.index(name)
                new_to_list.append(people_list[idx])
            elif email_exist_flag == 1: # email by itself
                person = ('f'+str(unknown_idx), 'l'+str(unknown_idx), email)
                new_to_list.append(person)
                unknown_idx = unknown_idx + 1
            else:
                person = (first_name, last_name, first_name + '_' + last_name)
                new_to_list.append(person)
        new_to_lists.append(new_to_list)
    return new-to_list
#--------------------------------------------------------
# Further processing of names: 
#  Adrian A. Jones ==> Adrian Jones (ignore middle names)
#  A. Adrian Johnes ==> Adrian Jones (remove short initials)
# Adrian Jones Sr. ==> Adrian Jones (remove Sr. Ph.D., III)
# Adrian. Jones ==> Adrian Jones (remove punctuation)
# 
# How to distinguish: 
# (('ANDREW J', 'BASCOM'), 'abascom@wradvisors.com'),
#  (('ANDREWJ', 'BASCOM'), 'abascom@wradvisors.com'),
# The second line does not have a space between ANDREW and J
# Check by removing spaces from first name and if there is a match, 
# keep the first name wihtout space

# After talking with Joey, I decided to only keep emails; each will 
#  will have a set of associated name (first, last). 
# Jr, Sr, PhD, etc will be removed from the names
#--------------------------------------------------------
def get_names_without_emails_from_list(lst):
    names_without_emails = []
    for k in lst:
        if not rex.match('.*@.*', k):
            names_without_emails.append(k)
    return names_without_emails
#--------------------------------------------------------
def get_names_without_emails_from_dict_tuples(dct):
    # Read a dictionary of tuples (name, email)
    # Return a list of tuples (field name, cleaned name)
    names_without_emails = []
    for k,v in dct.items():
        try:
            if not rex.match('.*@.*', v[1]):
                names_without_emails.append((k, v[0]))
        except:
            print(v)
    return names_without_emails
#--------------------------------------------------------
def compute_email_name_dicts1(field_dict):
    # each value of field dict is a 2-tuple: key, value
    # The key is the content of the raw email From:, To:, CC: fields
    # The value is the cleaned up email and value (not perfect cleanup)
    # Scan the values of field_dict, and create two dictionaries:
    #     name -> set of emails
    #     email -> set of names

    email_to_names = defaultdict(set)
    name_to_emails = defaultdict(set)
    for i,(k,v) in enumerate(field_dict.items()):
        email = v[1]
        name = v[0]
        email_to_names[email].add(name)
        name_to_emails[name].add(email)
    return email_to_names, name_to_emails
#--------------------------------------------------------
def compute_email_name_dicts(field_dict):
    # each value of field dict is a 2-tuple: key, value
    # The key is the content of the raw email From:, To:, Cc: fields (name+email in one string, dirty)
    # The value is the cleaned up email and name (not perfect cleanup)
    # Scan the values of field_dict, and create two dictionaries: 
    #     name -> set of emails
    #     email -> set of names
    # Both the key and values are clean

    # Also compute a list of clean names without emails and a list of clean names without emails. 
    
    email_to_names = defaultdict(set)
    name_to_emails = defaultdict(set)
    clean_names_without_email = set()
    clean_names_with_email = set()

    for i,(k,v) in enumerate(field_dict.items()):
        try:
            email = v[1]
            name = v[0]
        except:
            print(f"k,v: {k}____{v}")
            continue

        name_to_emails[name].add(email)

        if name and name != 'unrecognized':
            email_to_names[email].add(name)

    # Identify names without emails
    for name, v in name_to_emails.items():
        if '' in v:
            if len(v) == 1:
                clean_names_without_email.add(name)
            else:
                clean_names_with_email.add(name)

    clean_names_without_email = list(clean_names_without_email)
    clean_names_without_email.sort(key=lambda x: x)
    clean_names_with_email = list(clean_names_with_email)
    clean_names_with_email.sort(key=lambda x: x)

    return email_to_names, name_to_emails, clean_names_without_email, clean_names_with_email
#--------------------------------------------------------
def print_dict(dct, n=20, max_length=None):
    if max_length == None: 
        max_length = 100000
    print("len: ", len(dct))
    for i, (k,v) in enumerate(dct.items()):
        if len(v) > max_length: continue
        if i >= n: break
        print(f"{k}_______{v}")
#--------------------------------------------------------
def print_list(my_list, n=20):
    print("len: ", len(my_list))
    for i, el in enumerate(my_list):
        print(f"[{i}] : {el}")
        if i == n: break
#--------------------------------------------------------
def invert_dictionary(field_dict):
    # Transform a dictionary key_str => value_str into
    # an inverse dictionary value_str => set(key_str1, key_str2)
    unique = defaultdict(set)
    for k, v in field_dict.items():
        unique[v].add(k)
    return unique
#--------------------------------------------------------
def clean_to_unclean_names(field_dict):
    """
    field_dict: unclean => clean name, email)
    return: clean => set(unclean names)
    """
    clean_to_unclean = defaultdict(set)

    for k,v in field_dict.items():
        clean_to_unclean[v[0]].add(k)

    return clean_to_unclean
#--------------------------------------------------------
class StandardizeNames:
    def __init__(self, From, Cc, To, remove_if_longer_than=40):
        self.from_list = From
        self.cc_list = Cc
        self.to_list = To
        self.create_field_dict()
        self.remove_if_longer_than = remove_if_longer_than

    def clean_name(self, name):
        return clean_name(name)

    def create_field_dict(self):
        self.field_dict = create_field_dict(self.from_list, self.to_list, self.cc_list)

    def clean_field_dict_values(self):
        self.field_dict1, self.unique_names, self.removed, self.unrecognized_names = \
                clean_field_dict_values(self.field_dict, 
                remove_if_longer_than=self.remove_if_longer_than)

    def clean_to_unclean_names(self):
        """  string => set """
        self.clean_to_unclean = clean_to_unclean_names(self.field_dict1)
        #self.clean_to_unclean.sort(key = lambda x: x)
        # Given dict: str => set, sort the dictionary keys

    def compute_email_name_dicts(self):
        self.email_to_names, \
        self.name_to_emails, \
        self.clean_names_without_emails, \
        self.clean_names_with_emails = \
                compute_email_name_dicts(self.field_dict1)

    def get_names_without_emails_from_list(self, name_list):
        self.names_without_emails = get_names_without_emails_from_list(name_list)

    def get_names_without_emails_from_dict_tuples(self, name_dct):
        self.names_without_emails = get_names_without_emails_from_dict_tuples(name_list)

    def process(self):
        self.create_field_dict()
        self.clean_field_dict_values()
        self.compute_email_name_dicts()
        self.get_names_without_emails_from_list(self.field_dict1)

    def print_dict(self, dictionary, nb_to_print, max_length=10):
        print_dict(dictionary, nb_to_print, max_length=max_length)

    def print_list(self, my_list, nb_to_print):
        print_list(my_list, nb_to_print)

    def print_separator(self):
        print("========================================================================================" + 
              "==========================================")

    def print_data(self, nb_to_print=5, max_length=10):
        self.print_separator()
        print(">> field_dict <<")
        print("   Name => Name  (identity operator)")
        self.print_dict(self.field_dict, nb_to_print, max_length=max_length)

        self.print_separator()
        print(">> field_dict1 <<")
        print("   Name => (clean Name, email)")
        self.print_dict(self.field_dict1, nb_to_print, max_length=max_length)

        self.print_separator()
        print(">> email_to_names <<")
        print("   email => (sequence of clean names)")
        self.print_dict(self.email_to_names, nb_to_print, max_length=max_length)

        self.print_separator()
        print(">> names_to_emails <<")
        print("   name => (sequence of emails)")
        self.print_dict(self.name_to_emails, nb_to_print, max_length=max_length)

        self.print_separator()
        print(">> names_without_emails <<")
        print("   original full names ")
        self.print_list(self.names_without_emails, nb_to_print)

        self.print_separator()
        print(">> clean_names_without_emails <<")
        print("   list of cleaned names without no associated emails")
        self.print_list(self.clean_names_without_emails, nb_to_print)
        self.print_separator()

        self.print_separator()
        print(">> clean_names_with_emails <<")
        print("   list of cleaned names with associated emails")
        self.print_list(self.clean_names_with_emails, nb_to_print)
        self.print_separator()

    def name_matches(self, name_list):
        self.matches_df = nmlib.name_matches(name_list)
#--------------------------------------------------------
def clean_lowercase_name(name):
    name = rex.sub('sheilacos4gan', 'sheilacostigan', name)
    name = rex.sub('rudys4vers', 'rudystivers', name)
    name = rex.sub('marktarmeyat4mdesign', 'marktarmey@4mdesign', name)

    name = rex.sub('kris4ndozier', 'kristindozier', name)
    name = rex.sub('cris4nagarcia', 'cristinagarcia', name)
    name = rex.sub('jus4nvarn', 'justinvarn', name)
    name = rex.sub('dextermar4n', 'dextermartin', name)
    name = rex.sub('chris4nejwhite', 'christinejwhite', name)
    name = rex.sub('alsorren4', 'alsorrenti', name) # (Al Sorrenti)
    name = rex.sub('blairmar4n', 'blairmartin', name)
    name = rex.sub('richardsoncur4s', 'richardsoncurtis', name)

    name = rex.sub('dozierkris4n', 'kristindozier', name)
    name = rex.sub('gsquaredproduc4ons', 'gsquaredproductions', name)
    name = rex.sub('execu4veteam', 'executiveteam', name)
    name = rex.sub('christophercan4ello', 'christophercantiello', name)
    name = rex.sub('chris4hale', 'christihale', name)
    name = rex.sub('sarahvalen4ne', 'sarahvalentine', name)
    #name = rex.sub('chris_neece4re'   # NOT SURE Might be real. 

    return name
#--------------------------------------------------------
def update_header_list_of_lists(my_list, field_dict2, stand):
    count = 0
    count_nan = 0
    nb_exceptions = 0
    max_count = -1  # set to negative number to count all cases
    new_my_list = []
    break_flag = False
    # for lst in df.To.values:
    for m, lst in enumerate(my_list):
        new_split_list = set()
        if type(lst) != str:
            count_nan += 1
            new_my_list.append('invalid_nan')
            continue
        els = lst.split(";")
        for i, el in enumerate(els):
            count += 1
            try:
                f = field_dict2[el]
                if f == 'unrecognized':
                    f = 'invalid'
                # if a date
                elif rex.match(r'.*/20\d\d', el):
                    f = 'invalid'
                f = clean_lowercase_name(f)
                new_split_list.add(f)
            except:
                nb_exceptions += 1
                print(f"({i} Exception, el: __{el}__, ___field_dict1: {stand.field_dict1[el]}")
                print(f"  lst: __{lst}__")
                print(f"    field_dict2[{el}] = {field_dict2[el]}")
                print("     SHOULD NOT HAPPEN, otherwise new_my_list will be out of sync with my_list")
                traceback.print_exc()
                break_flag = True
                break
        # Rebuild semi-colon-separated list
        if break_flag == True:
            break
        new_my_list.append(";".join(new_split_list))
        if max_count > 0 and count >= max_count: break
    print("End count= ", count)
    print("nb nan: ", count_nan)
    print("Number exceptions: ", nb_exceptions)
    ### ERROR: new_my_list way too long. Not being updated at the right place
    print("len my_list, new_my_list: ", len(my_list), len(new_my_list))
    return new_my_list
#--------------------------------------------------------
def update_header_list(my_list, field_dict2, stand):
    count = 0   # set to negative number to count all cases
    max_count = -1  # set to negative number to count all cases
    nb_exceptions = 0
    new_my_list = []
    # for lst in df.To.values:
    print(len(my_list))
    for i, el in enumerate(my_list):
        if type(el) != str:
            # print(f"({i}) NOT string: {el}")  # nan
            new_my_list.append('invalid')
            continue
        try:
            f = field_dict2[el]
            if f == 'unrecognized':
                f = 'invalid'
            f = clean_lowercase_name(f)
            new_my_list.append(f)
            ### emily@oasis@comcast.net SHOULD BE CAPTURED. FIGURE THIS OUT.
            # print(f)
        except:
            nb_exceptions += 1
            print(f"({i} Exception, el: __{el}__, ___field_dict1: {stand.field_dict1[el]}")
            print(f"    field_dict2[{el}] = {field_dict2[el]}")
            print("     SHOULD NOT HAPPEN, otherwise new_my_list will be out of sync with my_list")
            traceback.print_exc()
            break
        if max_count > 0 and count >= max_count: break
        count += 1

    print("End count= ", count)
    print("Number exceptions: ", nb_exceptions)
    print("len my_list, new_my_list: ", len(my_list), len(new_my_list))
    return new_my_list
#--------------------------------------------------------
# MUST BE MODIFED WHEN DEALING WITH LISTS OF LISTS? Check into this.  <<<< 2022-03-12 (Saturday)
def construct_field_dict2(stand):
    field_dict2 =  {}
    for i, (k,v) in enumerate(stand.field_dict1.items()):
        if v == ():
            # print(f"k: {k}, empty value")
            field_dict2[k] = 'invalid'
        elif v[1] == '':
            field_dict2[k] = v[0]
        else:
            field_dict2[k] = v[1]
    return field_dict2
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
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
