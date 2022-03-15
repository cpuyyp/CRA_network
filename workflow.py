"""
# Execute multiple notebooks sequentially from a new notebook
## !jupyter nbconvert --execute --inplace execute_*.ipynb

!jupyter nbconvert --execute --inplace execute_new_preprocess.ipynb
!jupyter nbconvert --execute --inplace preprocessing_testing.ipynb
!jupyter nbconvert --execute --inplace sent_header.ipynb
!jupyter nbconvert --execute --inplace handle_to_header.ipynb
!jupyter nbconvert --execute --inplace standardize\ names.ipynb
!jupyter nbconvert --execute --inplace add_attributes_to_output.ipynb

# Test this
new_preprocess.ipynb  # not sure)
preprocessing_testing.ipynb  # (not sure)
handle_sent_header.ipynb
handle_to_header.ipynb
standadize\ names.ipynb

# also removes email duplicates
#   (smae From, Send, nb of chars, nb of words
# output: "output_with_attributes_no_duplicates"
# Also outputs: "output_with_stats_columns", which includes
# means and std of attribues (nb of chars/words) per Sender.
add_attributes_to_output.ipynb
"""

"""
TODO: 
    Use email_to_names and  name_to_emails to generate sets of names that are equivalent. 
    To each name, there is a set of emails. Names without emails might have more than one
    email associated with it. So: 
      1. how to assign names to emails? 
      2. how to assign an email to as many names as possible?`
"""

# 
import pandas as pd
import date_library as datelib
import numpy as np
import regex as rex
import process_to_library as tolib
from collections import defaultdict
from standardize_library import StandardizeNames
import name_matching_lib as nmlib

input_file = "output_0211.csv"
dates_file = "dates.csv"
date_output_file = "output_dates.csv"
to_cc_output_file = "output_to_cc.csv"
standard_names_output_file = "output_names.csv"
attributes_stats_output_file = "output_attr_stats.csv"
without_invalids_output_file = "output_without_invalid_mails.csv"

#================================================================================
### handle_sent_header.py
#--------------------------------------------------------------------------------

# File produced by Joey
df = pd.read_csv(input_file)
df = df.iloc[0:,:]

date = datelib.DateClass()
sent_lst = df['Sent'].values
date.normalize_dates(sent_lst)
date.create_dates_file(output_file=dates_file)
print("**********************************************************************")
print(f"===> Created {dates_file}")
date.update_output_file(df, date_output_file)
print("**********************************************************************")
print(f"===> Created {date_output_file}")

#================================================================================
### handle_to_Header
#--------------------------------------------------------------------------------

df = pd.read_csv(date_output_file)

cc_list = df['CC'].values   # 170 not processed out of 71,000
to_list = df['To'].values  # 566 not processed out of 71,000

new_cc_list, cc_not_processed = tolib.process_list(cc_list[0:], debug=False)
new_to_list, to_not_processed = tolib.process_list(to_list[0:], debug=False)
        
df["To"] = new_to_list
df["cc"] = new_cc_list
df.to_csv(to_cc_output_file, index=0)

#================================================================================
### standardize\ names.py
#----------------------------------------------------------------------------------

# Make sure I read in the original names in the mail list
# Each member of the Cc: and To: fields are triplets
# df = pd.read_csv('output_0211.csv')

df = pd.read_csv(to_cc_output_file)

stand = StandardizeNames(df, remove_if_longer_than=40)
stand.process()

stand.compute_email_name_dicts()

#my_list = stand.clean_names_with_emails + stand.clean_names_without_emails
#matches_df = nmlib.name_matches(my_list, cs_thresh=0.6, ngram=2)

stand.output_updated_headers(standard_names_output_file)
print("**********************************************************************")
print(f"===> Created {standard_names_output_file}")

#================================================================================
### add_attributes_to_output.ipynb
#----------------------------------------------------------------------------------

df = pd.read_csv(standard_names_output_file)
bodies = df.Body

nb_words = []
nb_chars = []
body_len = []
body_list = []

for i, row in enumerate(bodies):
    body = eval(row)
    body = " ".join(body)
    words = body.split(" ")
    text = "".join(words)
    body_list.append(body)
    body_len.append(len(body))
    nb_words.append(len(words))
    nb_chars.append(len(text))

df1 = df.copy()
df1['nb_words'] = nb_words
df1['nb_chars'] = nb_chars
df1['body_len'] = body_len
df1['body'] = body_list  # leaving original Body column
df1.to_csv(attributes_stats_output_file, index=0)
print("**********************************************************************")
print(f"===> Created {attributes_stats_output_file} with mail-level attributes")

# change Nan to empty strings
df2 = df1.copy()
df2 = df2.fillna('')

dfg = df2.groupby(["From", "To", "Sent", "body", "body_len", "nb_words"]).size().to_frame()

# Remove mail duplicates
df3 = df2.drop_duplicates(["From", "To", "Sent", "body"], keep='first')

# Now group emails by sender and calculate statistics
df3g = df3.groupby('From')

df3['mn_nb_words'] = df3g['nb_words'].transform('mean')
df3['std_nb_words'] = df3g['nb_words'].transform('std')
df3['mn_nb_chars'] = df3g['nb_chars'].transform('mean')
df3['std_nb_chars'] = df3g['nb_chars'].transform('std')
df3['email_count'] = df3g['nb_words'].transform('size')
df3.columns

# Confirm that the 'count' columns is the number of emails per sender
df3.groupby('From')['mn_nb_words'].size().values.sum(), df3.shape

df3.to_csv(attributes_stats_output_file, index=0)
print("**********************************************************************")
print(f"===> Created {attributes_stats_output_file} with sender-level statistics")

#================================================================================
### remove invalid rows
#----------------------------------------------------------------------------------

df = pd.read_csv(attributes_stats_output_file)

# Remove all rows with negative timestamps, used to indicate 
# an incorrect Sent field. 
df1 = df[df['timestamp'] > 0 ]

# Remove invalid rows 
df2 = df1[df1.From != 'invalid']

# Remove rows with empty From field (shows up as NaN)
df2 = df2[pd.isnull(df2.From) == False]
df2.shape

# CC elements to remove: 
# These are elements with characters other than [A-Za-z0-9] and [@]
# Not clear whether they come from. NOT REMOVED AS YET.
# * ['850)545-2095edward.kring@talgov.comsentfrommyiphonethanks'
# * "bryant ==> bryan  # remove quote)
# * "mike(firedept" ==> mike
# *  bellamy  # (what is this?)
# * cherie(planning => cherie   # why wasn't parenthesis removed?
#     * 51733_fn_34-5-ScottMaddox5_ln_54108.txt," Thomas, Debra (Planning) <Debra.Thomas@talgov.com"," Thursday, November 7, 2013 10:27 AM", Mayo      r & City Commissioners," Bryant, Cherie (Planning); McDonald,Earnest; Faris, Alison",, Palmer Avenue Block Party Ceremony Time, image001.      jpg,,False,False,True,False,False,"


def remove_invalid_from_list(my_list):
    # Construct new To: list with invalid cc: elements removed. Do not count them. 
    # Remove NaN as well.
    new_my_list = []
    for lst in my_list:
        split = lst.split(";")
        new_split = []
        for el in split:
            if not rex.match('.*invalid', el, flags=rex.I):
                new_split.append(el)
        new_my_list.append(";".join(new_split))
    return new_my_list


to_list = remove_invalid_from_list(df2.To.values)
cc_list = remove_invalid_from_list(df2.CC.values)
len(to_list), len(cc_list)


df3 = df2.copy()
df3.loc[:,'To'] = to_list
df3.loc[:,'CC'] = cc_list

df3.to_csv(without_invalids_output_file, index=0)

print("**********************************************************************")
print(f"===> Created {without_invalids_output_file}")
#=======================================================================================






