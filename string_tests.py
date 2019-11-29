
import re

"""
str =  'cabaide@aol.com, SteEva8@aol.com, TRHinson@aol.com, bill@billpeebles.com, "AmyHarllee" <office@billpeebles.com>, "Lamar Taylor" <ltaylor99@comcast.net>, "Will Messer"<wmesser@earlbacon.com>, thorntong@flalottery.com, drbilltucker@gmail.com,dsnewman77@gmail.com, "Kim Rivers" <kim@inkbridge.com>, "laurie hartsfield"<laurie.hartsfield@kccitallahassee.com>, christic@kingdomfirstrealty.com, kim@marpan.com,"Terrie Brooks" <Terrie@marpan.com>, llawler@moorebass.com, "Tom O\Steen"<tosteen@moorebass.com>, hmartin@nettally.com, wrb55@nettally.com,tsperry@oliverrenovation.com, "Jennifer Beerli" <jennifer@talcor.com>, murray@talcor.com'

str = ' gorodn, erlebach@'

pattern = r"(^[A-Za-z, ]+$)"
print("recipient_list= ", str)
m = re.search(pattern, str)
print("m= ", m)

pattern = r"\s*\w+\s*,\s*\w+\s*"
string = "gordon , fran <  "
m = re.search(pattern, string)
print("m= ", m)
"""

string = "Paige Carter ()"
print("removing junk: string= ", string)
string = re.sub(r"\(\)", "", string) 
print("after removal: ", string)

string = r"<gordon"
print("before: ", string)
string = re.sub(r"[<>\(\)\.\"]", "", string)
print("after: ", string)

string = r"xa0gordon\\xa0"
print("string= ", string)
string = re.sub(r"xa0|\\", " ", string) # removes all occurrences
print("string= ", string)

string= "Echo Kidd Gates, P.P.E.F. [mailto:]"
print("string= ", string)
# why does replacing " " by \b not working? I thought \b is white space. 
#string = re.sub(r"(,\s*[A-Z][\.]?)([A-Z][\.]?)+", "", string)

# works, but I need two-letter abbrev
#string = re.sub(r",(\s*[A-Z][\.]?)+", " ", string) 
string = re.sub(r",(\s*[A-Z][\.]?){2,}", " ", string)
print("string= ", string)

string = " P.E.   J. Keith Dantin"
print("string= ", string)
#re.sub(r"(\s*[A-Z][\.]?){2,}(\s+[A-Z]\.?)\s+(\w)\s+(\w)", r"", string)
#  " P.E.   J. Keith Dantin" ==> Keith J. Dantin
string = re.sub(r"(([A-Z][\.]?){2,})\s+([A-Z][\.]?)\s+(\w+)\s+(\w+)", r"\4 \3 \5", string)
print("string= ", string)


string = "gsd BCC: af ;lkjaf] af j]"
print("string= ", string)
string = re.sub(r"(BCC:.*?])", r"]", string)
print("string= ", string)

string="<mailto:kim@inkbridge.com>"
print("string= ", string)
string = re.sub(r"mailto:[ \w@\.]+", " ", string)
print("string= ", string)

string = "MikeWood in heaven"
print("string= ", string)
string = re.sub(r"\b([A-Z][a-z]+)([A-Z][a-z]+)\b", r"\1 \2", string)   # NOT TESTED
print("string= ", string)

