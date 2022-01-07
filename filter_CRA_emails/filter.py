import pandas as pd
import numpy as np

# 18,075 mails with empty subject
# 21,814 total emails

def handle_body(body):
    if (len(body) == 0):
        return ''
    elif (body[0:3] == 'Re:' or body[0:3] == 'RE:'):
        #print("return body")
        return body
    elif (body[-1] == ','):
        return ''
    elif (len(body) > 100):
        return ''
    else:
        return ''
#----------------------------------------
#filenm = "out7.csv"
filenm = "output7_new_sentiment.csv"
df = pd.read_csv(filenm)
df1 = df[['Subject','Body']]

bodies  = df1['Body'] # series
subjects = df1['Subject'] # series

body_list = []
for i in range(len(bodies)):
    bb = eval(bodies[i])
    if (len(bb) == 0):
        bb = ['','']  # No subject, empty message
    #body_list.append(bb[0])
    body_list.append(bb) # keep all the words (bb is a list)

subject_list = []
for i in range(len(subjects)):
    bb = eval(subjects[i])
    if (len(bb) == 0):
        bb = ['']
    subject_list.append(bb)

for i,row in enumerate(subject_list):
    print("=================================================================")
    if row[0] == '':
        # return '' or new subject
        subject = handle_body(body_list[i][0])
        row[0] = subject
        subject_list[i][0] = subject
        if (subject != ''):
            body_list[i] = body_list[i][1:] # NOT CORRECT
            #print("AFT sub: %s____________%s" % (subject_list[i][0], body_list[i][0:50]))
            #print("%s" % body_list[i][0:50])
        else: 
            print("(empty subj), body:  %s" % body_list[i][0:50])  # empty subject
            pass
            #print("BEF sub: %s_______%s" % (row, body_list[i][0:50]))
    else:
        subject_list[i][0] = row[0]

