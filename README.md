# CRA network code

## Preprocessing (Updated Feb 11, 2022)

Step 1: Splitting emails

By this time, you should have all text files from https://data.tallahassee.com/fbi-cra-documents/ ready in a folder.

"preprocessing_testing.ipynb" will identify individual emails from a collection of multiple email text, save them as individual files to another folder.

Step 2: Splitting emails

"building csv.ipynb" reads each individual files generated by the previous step, identify headers, and build a csv.

Step 3: Name standardization

There are missing labels among (From, Sent,and  emails), and some clean up is necessary. Each person is stored as a triplet (first name, last name, email), 
	  extracted from various fields. If a name component is not found, it is generated. Assigned first names become "f1", "f2", etc. Assigned last names
	  are "l1", "l2", etc. Assigned emails are "first.last" (we will and "@fake.com" sign in the near future. 
This triplet forms a unique id.

The standardization code, "standardize names.ipynb" reads the output of "building csv.ipynb" and produces a new csv file ("output_date_name_standardized.csv").

Step 4: Data enhancement. Run the notebook "add_attributes_to_output.ipynb" to add statistical information to the csv file. 
This program produces the file "output_with_stats_columns.csv", which includes character and word count per email, along with 
averages and standard deviations across mailers. Note the presents of outliers, with several 1000 characters. Wehn calculating embeddings, these might
have to be removed to avoid skewing the data. 

## Input data information

Using 'output4.csv' as the inputing data, in which emails are sorted in time ascending order.

## File desciption

### function_library.py

'function_library.py' contains all the functions that are used.

### new_preprocess.py

'new_preprocess.py' is the file that process 'output4.csv'. The first pass drops several useless columns from the original dataframe for efficiency. The second pass will standardize names to be triplets in the form of (first_name, last_name, email_address). For some special cases, where at least one of the triplet is missing, a fake name or fake email address will be assigned.

### handle_sent_header.ipynb
Read an output.csv file with a Sent: header, and generate consistent formatting for all the entries. Generate
three new columns: Date (yyyy-mm-dd), Time (hh-mm), and timestamp (the number of seconds from a fixed time around 1970. 
All times are expressed in Eastern Daylight Time. I ignore the one hour difference between Eastern Daylight Time (EDT)
and Eastern Savings Time (EST) since the two types of time cannot occur on the same day. Also, office hours are 8-5 pm 
typically in both zones in different parts of the year. However, 5 am  PST is converted to 8 am EST. Spelling errors in the 
different elements have been corrected. 

### handle_to_header.ipynb  (handling the Cc header is similar to the to: header)
Handle the comma separated elements of from_header and transform them to semi-colon separated fields. 
Treat various special cases, the most prominent being when the number of fields is the same as the number of commas. In 
that case, simply replace the commas by semi-colons. 

TODO:
1. add a column to the dataframe to show whether the email is sent in day time or night time
2. calculate some stats about the length of title and subject, and add to the dataframe as columns
3. generate a sender/recipient by year table, sort the table by the total number of emails in the row.

### network_graph.py

'network_graph.py' reads the pickle file generated by 'new_preprocess.py', create connection matrix, create id2name dictionary using the sorted unique name list. In the end, the plot is generated and saved into pdf.

### Some old/testing codes

Old codes include: networkx.ipynb, networkx_ge.py, preprocess.py

Testing codes include: string_tests.py

All can be cleaned later.

### Additional files
- build_new_output_csv.ipynb : create an output file, starting from individual emails. File does not use any local function library.
- preprocessing_testing.ipynb : Read individual text files, and create collection of emails

#-------------------------------------------------------------

Column descriptions of the file output_with_stats_columns.csv . 
Each row corresponds to an email. Duplicates have been removed. 
Two mails are duplicates of each other if the following fields match: 
'From', 'To', 'Sent', 'Body' . This is probably not foolproof. 
We started with 71,143 emails, and have 39,444 emails after removal of 
duplicates. Note that these emails were printed to pdf by the FBI, and 
we then translated the pdf to ascii. There are quite a few errors, with many 
fields missing. However, we did not access to the original emails. 

A collection of pdfs were translated to text files (some available at Tallahassee
Democrat). A text file might have a name such as: "18-3-Inkbridge2013-1-0.txt" . 
These files are broken up into  individual emails, with all headers written in a standard
order (which is not the case in emails), and with several additional headers to characterize
the email more precisely. We generated 71,143 emails. Each email is stored in a separate
file. A typical file name might be: "23793_fn_18-3-Inkbridge2013-1-0_ln_130.txt". The first 
five digits constitute the email id, ranging from  0 to 71,142. Next is the name of the 
pdf file that contains the email, prefixed by "fn", and finally, the line number that 
corresponds to the last line of the email, prefixed by "ln". 

filenm: string
	The filename that contains the email at this row. 

From: string that encloses a tuple of three strings
	The mail originator (i.e., sender). It is always a single person. Sometimes there is more information than 
	the sender, which indicates some error in the conversion from text to mail. The process is 
	imperfect. 

Sent: string that contains a 3-tuple of strings
    The time at which the email was sent. This must be converted to a proper datetime object, such as 
	the number of seconds from a specific point in time. Note that all times should have the same frame
	of reference. In this field, one finds times in multiple time zones, such as EST, GST, PST, etc. 
	The time registration is a task that remains to be done. Sent is a tuple from from  first name, last 
	name, and email.

To: string
    One more more people receiving the email. 
	Each person is stored as a tuple (first, last, email). The string encloses a list of tuples. 

CC: string
    One or more recipients cc'ed on the email. 
	Each person is stored as a tuple (first, last, email). The string encloses a list of tuples. 

Bcc: string
	One or more recipients who are BCCed. Not clear whether BCC'ed recipieents are always shown. 
	Each person is stored as a tuple (first, last, email). The string encloses a list of tuples. 

Subject: string
   Subject of the email. If the subject is more than 80 characters, it is probably due to an error when 
   generating the table. 

Attachments: string
   List of file names,  usually separated by a semi-colon.

Importance: string
   Whether the email is important or not. Values are "High" or an empty string. 

isThread: Boolean
   Whether the email is part of a thread or not: True/False.

isAutoMessage: Boolean
   Whether the email was generated automatically, e.g., by Google Calendar (True/False)

isDisplacement:  Boolean
	Whether the mail has a displacement section. A displacement (our team's term) refers to two or more successive headers 
	with empty content, followed by the content. Example: 

	   From:
       Sent:
       To:
	   Joe Johnson
	   Wed. Jan. 4, 2022
	   Frank Williams

hasAllCapLine: Boolean
	Whether the email contains headers that are all caps, which suggests and internal memo is contained within the email.

hasBadDate: Boolean
	Whether a badly formatted date is contained in the email. <<< Please Check on this, Joey

Body: string 
   The email body, as a list of strings. Each string is a line in the pdf file. 
   The entire list is enclosed in a string. Use Python's `eval` to transform the string into an array

nb_words: Integer
   The number of words in the body. Elements `body` columns are split at spaces. Thus, words are separated by 
   spaces. Punctuation is not handled. Thus, "However, the boy", is split into three words: "However,", "the", and 
   "boy". Note the comma is part of the first word. Another approach would be to isolate the punctuation and consider 
   a comma as a word. There are multiple approaches to consider. 

nb_chars:  Integer
   The words used to compute `nb_words` are joined with no spaces, and the resulting character is counted. The number of 
   characters defineds `nb_chars`. Therfore, spaces in the text do not count towards the character acccount. Other approaches
   are possible, nad possibly desirable.

body: string
	The entire email body as a string, obtained by joining the elements of Body, separated by a space. Multiple spaces are maintained. 
	Empty lines are maintained. Newlines have not been removed. 

body_len:  Integer
	The length of `body`. It is clear that `body_len` is always greater than `nb_chars'. This surplus averages to about 20%. . 

Error_from:  Boolean
	True when the length of (first,last,email) exceeds 200 characters. This indicates high probability of an error in the `From` field. 

Error_sent: Float
	True when the length of Sent field exceeds 200 characters. This indicates high probability of an error in th	 `Sent` field. 

email_count: Integer
	The number of emails for each unique sender. 

mn_nb_words: Float
	The average number of words in an email for each individual sender. 

std_nb_words: Float
	The standard deviation of the number of words in an email for each individual sender. 

mn_nb_chars: Float
	The average number of characters in an email for each individual sender. 

std_nb_chars: Float
	The standard deviation of the number of characters in an email for each individual sender. 

date_sent: string
	The day the email was sent expressed as 'yyyy-mm-dd'

time_sent: string
    The time the email was sent expressed as 'hh-mm'

timestamp: float
	The number of seconds measured from a fixed reference point, somewhere in the 1970's. This 
	will allow temporal algorithms to be applied easily.

#-------------------------------------------------------------
