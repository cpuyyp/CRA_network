import pandas as pd

def flattenListOfLists(lis):
    """
    Unpacking list of lists.

    :param lis: List of lists
    :type people_list: list(list())

    :return: Return the flattened list
    :rtype: list()
    """
    return [item for sublist in lis for item in sublist]

def readEmailsFromFile(filename):
    """
    Read emails from records. Correct the senders and receivers
    with the correct type.

    :param file: Name of the csv file.
    :type people_list: str

    :return: Return the email records in a dataframe.
    :rtype: DataFrame
    """
    df_email = pd.read_csv(filename, index_col = 0)
    from_list = df_email['From'].values.tolist()
    for i in range(len(from_list)):
        from_list[i] = eval(from_list[i])
    df_email['From'] = from_list

    to_list = df_email['To'].values.tolist()
    for i in range(len(to_list)):
        to_list[i] = eval(to_list[i])
    df_email['To'] = to_list

    cc_list = df_email['CC'].values.tolist()
    for i in range(len(cc_list)):
        cc_list[i] = eval(cc_list[i])
    df_email['CC'] = cc_list

    return df_email
