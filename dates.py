from dateutil import parser
import datetime
import date_library as datelib
import pytz

tzinfo = {
          "EST": pytz.timezone('US/Eastern'), #-4,
          "EDT": pytz.timezone('US/Eastern'), #-4,
          "CDT": pytz.timezone('US/Central'),
          "MDT": pytz.timezone('US/Mountain'),
          "PDT": pytz.timezone('US/Pacific'),  #-7
          "HDT": pytz.timezone('US/Hawaii'),  #-7
         }

dates = []
dates.append("Friday, May 13, 2016 10:13 AM EST")
dates.append("Friday, May 13, 2016 10:13 AM ")
dates.append("Friday, May 13, 2016 10:13 AM EDT")  # EDT works
dates.append("March 26, 2017 at 9:21:19 AM EDT")
dates.append(" Tuesday, May 28, 2013 2:38 PM")
dates.append("2/3/17 1:29 PM (GMT-05:00)")

# E.g.: Eastern Daylight Time: the international meetings are held
# between 10 am to 6 pm whereas for Eastern Standard Time the same meeting and conference
# are held between 9 am to 5 pm.  So, 10 am EDT  <=====> 9 am EST


for i, date in enumerate(dates):
    print("date: ", date)
    nm = parser.parse(date, tzinfos=tzinfo)
    nm1 = nm.astimezone(pytz.utc)
    time_stamp = nm1.timestamp()
    time_stamp -= 5 * 3600
    #if i == 0:
        #time_stamp -= 3600
    #elif i == 1:
        #time_stamp -= 0
    new_time = datelib.timestampToDateTimeUTC(time_stamp)
    print(date,"_________", nm, "_________", nm1, "______", new_time)


