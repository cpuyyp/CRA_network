#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from datetime import date, datetime
import time 

from IPython.display import display

#------------------------------------------------------
def timestampToDateTimeUTC(ts):
    # ts: timestamp in seconds
    #dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    dt_tm = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
    return dt_tm[0:10], dt_tm[11:16]
#---------------------------------------------------------
def timestampToDateTimePTY(ts):
    # ts: timestamp in seconds
    dt_tm = timestampToDateTimeUTC(ts + 5*3600)
    return dt_tm
#---------------------------------------------------------
def dateTimeUTCToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0000"

    # convert string to datetimeformat
    date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#---------------------------------------------------------
def dateTimePTYToTimestamp(date='2019/10/01', tim='20:00'): #, UTC_offset='0'):
    timestamp = 0;
    date = date + tim + "+0500"  # PTY is 5 hrs ahead of UTC

    # convert string to datetimeformat
    date = datetime.strptime(date, "%Y/%m/%d%H:%M%z")

    # convert datetime to timestamp
    return int(datetime.timestamp(date))
#------------------------------------------------------------
def to_datetime(series):
    series_tmz = series.str[-12:-7]
    series_dtz = pd.to_datetime(series)
    return pd.concat([series_dtz, series_tmz], axis=1)
#------------------------------------------------------
def Zulu2PTY(datetime):
    """
    Convert from Zulu to PTY date-time.

    Parameters: 
        datetime [Series]: a series of date/times stored in nanoseconds integer format

    Return: 
        A series of date/times in nanoseconds integer format in Panama city

    If it is 1:00 pm Zulu, it is 6:00 pm PTY time (DTML)
    """

    # PTY time is earlier than Zulu time by 5 hours. However, data from the web suggests 
    # that PTY is 5 hours later than Zulu, and even 4 hours later according to one site. I have no idea what the truth is. 
    # According to Miguel, Panama is always 5 hours BEHIND Zulu

    return datetime - 5 * 3600 * 1000000000 # 5 hours difference # clutching at straws
#--------------------------------------------------
