from datetime import datetime
from tqdm import tqdm_notebook, tnrange, notebook
import re
#-------------------------------------------------------------------------------------------------
def process_dates(df):
    dates = df["Sent"].values.tolist()
    for i in range(len(dates)):
        dates[i] = eval(dates[i])
    new_dates = []
    for idx, date in enumerate(tqdm_notebook(dates)):
        if date == []: continue
        # if date != []:
        dt = date[0]
        print("dt: ", dt)
        try:
            new_date = datetime.strptime(dt, "%A, %B %d, %Y %I:%M %p")
        except:
            try:
                new_date = datetime.strptime(dt, "%A, %B %d, %Y %I: %M %p")
            except:
                try:
                    new_date = datetime.strptime(dt, "%A %B %d, %Y %I:%M %p")
                except:
                    try:
                        new_date = datetime.strptime(dt, "%B %d, %Y %I:%M:%S %p %Z")
                    except:
                        try:
                            new_date = datetime.strptime(
                                dt, "%B %d, %Y, %I:%M:%S %p %Z"
                            )
                        except:
                            try:
                                new_date = datetime.strptime(
                                    dt, "%B %d, %Y %H:%M:%S %Z"
                                )
                            except:
                                try:
                                    new_date = datetime.strptime(
                                        dt, "%A. %B %d, %Y %I:%M %p"
                                    )
                                except:
                                    try:
                                        new_date = datetime.strptime(
                                            dt, "%B %d, %Y %I:%M %p"
                                        )
                                    except:
                                        try:
                                            new_date = datetime.strptime(
                                                dt, "%A. %B %d. %Y %I:%M %p"
                                            )
                                        except:
                                            try:
                                                new_date = datetime.strptime(
                                                    dt, "%A , %B %d, %Y %I:%M %p"
                                                )
                                            except:
                                                try:
                                                    new_date = datetime.strptime(
                                                        dt, "%A, %B %d. %Y %I:%M %p"
                                                    )
                                                except:
                                                    try:
                                                        new_date = datetime.strptime(
                                                            dt,
                                                            "%A, %B %d, %Y %I:%M:%S %p",
                                                        )
                                                    except:
                                                        try:
                                                            new_date = datetime.strptime(
                                                                dt,
                                                                "%A, %B %d, %Y %I:%M:%S %p Eastern Time (US Canada)",
                                                            )
                                                        except:
                                                            try:
                                                                new_date = datetime.strptime(
                                                                    dt,
                                                                    "%A, %B %d, %Y %I:%M",
                                                                )
                                                            except:
                                                                try:
                                                                    new_date = datetime.strptime(
                                                                        dt,
                                                                        "%B/%d/%Y %I:%M %p",
                                                                    )
                                                                except:
                                                                    try:
                                                                        new_date = datetime.strptime(
                                                                            dt,
                                                                            "%B %d, %Y at %I:%M %p",
                                                                        )
                                                                    except:
                                                                        try:
                                                                            new_date = datetime.strptime(
                                                                                dt,
                                                                                "%A, %B %d, %Y at %I:%M %p",
                                                                            )
                                                                        except:
                                                                            try:
                                                                                new_date = datetime.strptime(
                                                                                    dt,
                                                                                    "%A, %B %d, %Y %X %p (UTC-05:00) Eastern Time",
                                                                                )
                                                                            except:
                                                                                try:
                                                                                    new_date = datetime.strptime(
                                                                                        dt,
                                                                                        "%A, %B %d, %Y %X %p (UTC-05:00) Eastern Time (US & Canada)",
                                                                                    )
                                                                                except:
                                                                                    new_date = datetime.strptime(
                                                                                        "00:00:00",
                                                                                        "%X",
                                                                                    )
                                                                                    print(
                                                                                        "1",
                                                                                        idx,
                                                                                        dt,
                                                                                    )
                                        #                                 Wednesday, February 25, 2015 at 9:12 AM
                                        #                                 Friday, February 24, 6:47 PM
                                        #                                 Sunday, March 22, 2015 7:48:38 PM (UTC-05:00) Eastern Time
                                        #                                 Friday, August 05, 2016 5:34:39 PM (UTC-05:00) Eastern Time (US & Canada)
        else:
            new_date = datetime.strptime("00:00:00", "%X")
            print(idx, dt)
        
        new_dates.append(new_date)
    return new_dates
#-------------------------------------------------------------------------------------------------
def process_dates_new(df):
    """
    Transform time email sent into a standardized format
    
    Parameters
    ----------
    df : Pandas dataframe with emails and associated metadata
    
    Return
    -------
    dates : date column
    """
    dates = df["Sent"].values.tolist()
    for i in range(len(dates)):
        dates[i] = eval(dates[i])
    new_dates = []
    for idx, date in enumerate(tqdm_notebook(dates)):
        if date == []: continue
        dt = date[0]
        # print("dt: ", dt)
        dt1 = re.sub(r"Mon.*?,|Tue.*?,|Wed.*?,|Thu.*?,|Fri.*?,|Sat.*?,|Sun.*?,", ",", dt)
        dt1 = re.sub(',+', ' ', dt1)
        dt1 = re.sub('\.+', ' ', dt1)
        dt1 = re.sub('/+', ' ', dt1)
        dt1 = re.sub('at',' ', dt1)
        dt1 = re.sub(' +', ' ', dt1)
        dt1 = re.sub(': ', ':', dt1)
        # dt = dt1
        print("dt1: ", dt1)

        # print("dt1= ", dt1)
        # Transform all commas into spaces
        
        # Transform all double spaces into single spaces
        try:
            new_date = datetime.strptime(dt, "%A %B %d %Y %I:%M %p")
        except:
            try:
                new_date = datetime.strptime(
                    dt, "%B %d %Y %I:%M:%S %p %Z"
                )
                print("==> %I:%M:%S %p %Z, dt: ", dt)
            except:
                try:
                    new_date = datetime.strptime(
                        dt, "%B %d %Y %H:%M:%S %Z"
                    )
                    print("==> %Z format, dt: ", dt)
                except:
                    try:
                        new_date = datetime.strptime(
                            dt, "%B %d %Y %I:%M %p"
                        )
                    except:
                        try:
                            new_date = datetime.strptime(
                                dt,
                                "%A %B %d %Y %I:%M:%S %p",
                            )
                        except:
                            try:
                                new_date = datetime.strptime(
                                    dt,
                                    "%A %B %d %Y %I:%M:%S %p Eastern Time (US Canada)",
                                )
                            except:
                                try:
                                    new_date = datetime.strptime(
                                        dt,
                                        "%A %B %d %Y %I:%M",    ##### There is no AM/PM. How can we know the time? %I is 12h clock
                                    )
                                    print("==> no am/pm, dt: ", dt)
                                except:
                                    try:
                                        new_date = datetime.strptime(
                                            dt,
                                            "%B %d %Y %I:%M %p",
                                        )
                                    except:
                                        try:
                                            new_date = datetime.strptime(
                                                dt,
                                                "%A %B %d %Y %I:%M %p",
                                            )
                                        except:
                                            try:
                                                new_date = datetime.strptime(
                                                    dt,
                                                    "%A %B %d %Y %X %p (UTC-05:00) Eastern Time",
                                                )
                                            except:
                                                try:
                                                    new_date = datetime.strptime(
                                                        dt,
                                                        "%A %B %d %Y %X %p (UTC-05:00) Eastern Time (US & Canada)",
                                                    )
                                                except:
                                                    new_date = datetime.strptime(
                                                        "00:00:00",
                                                        "%X",
                                                    )
                                                    # print(
                                                    #     "1",
                                                    #     idx,
                                                    #     dt,
                                                    # )
                                        #                                 Wednesday, February 25, 2015 at 9:12 AM
                                        #                                 Friday, February 24, 6:47 PM
                                        #                                 Sunday, March 22, 2015 7:48:38 PM (UTC-05:00) Eastern Time
                                        #                                 Friday, August 05, 2016 5:34:39 PM (UTC-05:00) Eastern Time (US & Canada)
        else:
            new_date = datetime.strptime("00:00:00", "%X")
            print(idx, dt)
        
        new_dates.append(new_date)
    return new_dates
#-----------------------------------------------------------------------