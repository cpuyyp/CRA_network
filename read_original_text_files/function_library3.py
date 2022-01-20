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
def standardize_date_string(line):
    #dt1 = re.sub(r"Mon.*?,|Tue.*?,|Wed.*?,|Thu.*?,|Fri.*?,|Sat.*?,|Sun.*?,", ",", line)
    # Cannot have unconverted characters or spaces ending line
    dt1 = re.sub(r"Date:", "Sent:", line)
    dt1 = re.sub(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun).*?,", ",", dt1)
    #print(" 1 dt1= ", dt1)
    #dt1 = re.sub(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun).*?,", r"x", line)
    #print(" 2 dt1= ", dt1)
    dt1 = re.sub(',+', ' ', dt1)
    #print(" 3 dt1= ", dt1)
    dt1 = re.sub('\.+', ' ', dt1)
    #print(" 4 dt1= ", dt1)
    dt1 = re.sub('/+', ' ', dt1)
    #print(" 5 dt1= ", dt1)
    dt1 = re.sub(r'\bat\b',' ', dt1)
    #print(" 6 dt1= ", dt1)
    dt1 = re.sub(' +', ' ', dt1)
    #print(" 7 dt1= ", dt1)
    dt1 = re.sub(': ', ':', dt1)
    dt1 = re.sub('\+0000', '', dt1)
    # Remove seconds: 1:43:05 ==> 1:43
    dt1 = re.sub('(:[0-9]{1,2}):[0-9]{1,2}', r"\1", dt1)
    dt1 = re.sub('EDT.*', '', dt1)
    #print(" 8 dt1= ", dt1)
    #"""
    return dt1.strip()
#-------------------------------------------------------------------------------------------------
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
                    dt, "%B %d %Y %I:%M:%S %p %Z"  # 12 hrs
                )
                print("==> %I:%M:%S %p %Z, dt: ", dt)
            except:
                try:
                    new_date = datetime.strptime(
                        dt, "%B %d %Y %H:%M:%S %Z"  # 24 hrs
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
def process_dates_new_string(date_str):
    """
    Transform time email sent into a standardized format
    
    Parameters
    ----------
    date_str : string with date
    
    Return
    -------
    datetime object
    """
    #print("date_str: ", date_str)
    for idx, date in enumerate([date_str]):
        idx = 0
        date = date_str
    
        if date == []: continue
        dt = date
        #print("dt: ", dt)
        dt1 = re.sub(r"Date:", "Sent:", dt)
        dt1 = re.sub(r"'iuesday", "Tuesday", dt1)
        dt1 = re.sub(r"Februaiy", "February", dt1)
        dt1 = re.sub(r"Februaly", "February", dt1)
        dt1 = re.sub(r"(\w*?) ([0-9]) ([0-9])", r"\1 \2\3", dt1)
        dt1 = re.sub(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun).*? ", " ", dt1)
        dt1 = re.sub(',+', ' ', dt1)
        dt1 = re.sub('\.+', ' ', dt1)
        dt1 = re.sub('/+', ' ', dt1)
        #print("before at, dt1= ", dt1)
        dt1 = re.sub(r'\bat\b', '', dt1)  # remove 'at'
        #print("after at, dt1= ", dt1)
        dt1 = re.sub(' +', ' ', dt1)
        dt1 = re.sub(': ', ':', dt1)
        # Remove seconds: 1:43:05 ==> 1:43
        dt1 = re.sub('(:[0-9]?):[0-9]?', r"\1", dt1)
        # EDT: Eastern Daylight time: 4 hours behind Coordinated Universal Time
        # EST: Eastern Standard time: 5 hours behind Coordinated Universal Time
        dt1 = re.sub(r'\s*?EDT.*', '', dt1)
        # 19 Mar 2015 15:20:50 EST
        dt1 = re.sub(r'\s*?EST.*', '', dt1)
        dt1 = re.sub(r'\s*?Eastern Standard Time.*', '', dt1)
        dt1 = re.sub(r'\s*?Eastern Daylight Time.*', '', dt1)
        dt1 = re.sub(r'\s*?Eastern Time.*', '', dt1)
        dt1 = re.sub(r'\s*?CDT.*', '', dt1)
        dt1 = re.sub(r'\s*?\(GMT.*', '', dt1)
        dt1 = re.sub(r'\s*?GMT.*', '', dt1)
        dt1 = re.sub(r'\s*?PDT.*', '', dt1)
        #print("before: dt1: ", dt1)
        dt1 = re.sub(r' *\(UTC.*', '', dt1)  # This line does not always work. WHY? 
        #print("after: dt1: ", dt1)
        dt1 = re.sub(r'\s*?\(US & Canada.*', '', dt1)
        dt1 = re.sub(r'\s*?US Mountain.*', '', dt1)
        dt1 = re.sub(r'([0-9])\s:([0-9]{1,2}\s(AM|PM))', r'\1:\2', dt1)

        # ASK JOEY
        # The string "November 8 2016 10:41 AM (UTC-05:00) Eastern Time (US & Canada)"  is not handle correctly with 
        #      dt1 = re.sub(r'\s*?\(UTC.*', '', dt1)  # WHY NOT? 
        # file_nb = 11, 9-1-Adam-Corey-2012-1-0.txt
        # 

        #print("   dt1: ", dt1)
        dt = dt1.strip()  # NEEDED
        #print("dt1: ", dt1)
    
        #print("dt1xx= ", dt1)
        # Transform all commas into spaces

        """
        %A: full weekday name
        %B: Full month name
        %b: Abbreviated month name
        %d: Minute (zero-padded)
        %d: day of month
        %Y: Year, 4 digits
        %I: Hour (12-hour clock)
        %S: Seconds, zero-padded
        %p: AM or PM
        %Z: Time zone name
        """

        # Transform all double spaces into single spaces
        try:
            new_date = datetime.strptime(
                dt, "%B %d %Y %I:%M:%S %p %Z"  # 12 hrs
            )
            #print("==> %I:%M:%S %p %Z, dt: ", dt)
        except:
            try:
                #print("try: ", dt)
                new_date = datetime.strptime(
                    dt, "%B %d %Y %H:%M:%S %Z"  # 24 hrs
                )
                #print("==> %Z format, dt: ", dt)
            except:
                try:
                    #print("try ", dt)
                    new_date = datetime.strptime( dt, "%B %d %Y %I:%M %p")
                except:
                    try:
                        #print("try ", dt)
                        new_date = datetime.strptime( dt, "%d %b %Y %H:%M")
                    except:
                        #print("except it ...")
                        try:
                            #print("try: ", dt)
                            new_date = datetime.strptime( dt, "%B %d %Y %I:%M %p")
                        except:
                            try:
                                new_date = datetime.strptime( dt, "%b %d %Y %I:%M %p")
                            except:
                                try:
                                    # 12 26 2014 08:32 AM  (month day 2014 ...)
                                    #print(f"month day year ...{dt}...")
                                    new_date = datetime.strptime(dt, "%m %d %Y %I:%M %p")
                                except:
                                    try:
                                        new_date = datetime.strptime(dt, "%b %d %Y %H:%M")
                                    except:
                                        try:
                                            new_date = datetime.strptime(dt, "%B %d %Y %H:%M")
                                        except:
                                            try: 
                                                new_date = datetime.strptime(dt, "%B %d %Y")
                                            except:
                                                try:
                                                    new_date = datetime.strptime(dt, "%B %d")
                                                except:
                                                    try:
                                                        new_date = datetime.strptime(dt, "%m %d %Y")
                                                    except Exception as err:
                                                        #new_date = datetime.strptime(
                                                            #"00:00:00",
                                                            #"%X",
                                                        #)
                                                        #print("Last exception, arg date_str: ", date_str)
                                                        print(f"... dt: ...{dt}...")
                                                        #print(Exception, err)
                                                        print("==================================")
                                                        return "0000000"
                                                        # print(
                                                        #     "1",
                                                        #     idx,
                                                        #     dt,
                                                        # )
                                                        #             Wednesday, February 25, 2015 at 9:12 AM
                                                        #             Friday, February 24, 6:47 PM
                                                        #             Sunday, March 22, 2015 7:48:38 PM (UTC-05:00) Eastern Time
                                                        #             Friday, August 05, 2016 5:34:39 PM (UTC-05:00) Eastern Time (US & Canada)
    return new_date
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
