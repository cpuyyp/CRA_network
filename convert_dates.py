# From https://www.djangosnippets.org/snippets/995/

import pytz
import dateutil.parser

TZINFOS = {
    'PDT': pytz.timezone('US/Pacific'),
    # ... add more to handle other timezones
    # (I wish pytz had a list of common abbreviations)
}

datestring = '11:45:00 Aug 13, 2008 PDT'

# Parse the string using dateutil
datetime_in_pdt = dateutil.parser.parse(datestring, tzinfos= TZINFOS)

# t is now a PDT datetime; convert it to UTC
datetime_in_utc = datetime_in_pdt.astimezone(pytz.utc)

# Let's convert it to a naive datetime object
datetime_naive = datetime_in_utc.replace(tzinfo = None)
