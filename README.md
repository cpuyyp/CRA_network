# CRA network code

## Preprocessing (Updated Feb 11, 2022)

Step 1: Splitting emails

By this time, you should have all text files from https://data.tallahassee.com/fbi-cra-documents/ ready in a folder.

"preprocessing_testing.ipynb" will identify individual emails from a collection of multiple email text, save them as individual files to another folder.

Step 2: Splitting emails

"building csv.ipynb" reads each individual files generated by the previous step, identify headers, and build a csv.

## Input data information

Using 'output4.csv' as the inputing data, in which emails are sorted in time ascending order.

## File desciption

### function_library.py

'function_library.py' contains all the functions that are used.

### new_preprocess.py

'new_preprocess.py' is the file that process 'output4.csv'. The first pass drops several useless columns from the original dataframe for efficiency. The second pass will standardize names to be triplets in the form of (first_name, last_name, email_address). For some special cases, where at least one of the triplet is missing, a fake name or fake email address will be assigned.

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
