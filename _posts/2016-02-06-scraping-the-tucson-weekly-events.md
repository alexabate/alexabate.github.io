---
layout: post
title: "Scraping the Tucson Weekly events"
tags:
    - python
    - notebook
--- 
## Web scraping in the mire

I wanted to write a fairly simple web app that would provide a "value added"
search of Tucson event listings in the Tucson Weekly. A simple search of events
occuring over a weekend returns hundreds of results to potentially sift through,
and personally I'd like to filter out certain things like weekly drum circles
and listings of an ongoing months-long exhibition in a generic art gallery.

Obviously the website allows you to do some simple filtering, but I'm going to
see if I can tune it up.

There is this great [blog post](http://nycdatascience.com/nyc-concert-data-web-
scraping-with-python/) that demos exactly how to start this kind of project
using BeautifulSoup via an example of scrapting the concert listings from NYC's
Village Voice.

# First task: simple scrape of event data into a table

Later on, when this is approaching an initial version of the actual app, this
will be turned into a weekly scheduled job that grabs the event data from the
current date until some future date (maybe a month?) and stores it.

Crawling over the pages returned for a particular date, starting from the
current date, and going up to some N days later is straight forward. All that
was needed was minor modifications to the python code from the blog post.

However, unlike the Village Voice music listings the html `div` tags used by the
Tucson Weekly website are often a lot more obscure (in my html-newbie opinion).
After sifting through the html I found that the `div` tag labelled "EventListing
clearfix" was the one that corresponding to a single event. 

**In [17]:**

{% highlight python %}
import datetime
import requests
from bs4 import BeautifulSoup

 
# just do current day for illustration
numDays = 1
for j in range(0, numDays):

    nEvents = 0
    
    # get string with date
    date = datetime.date.today() + datetime.timedelta(days = j) 
    dateString = date.strftime("%Y-%m-%d")
    print "DATE:"
    print dateString
    
    
    baseUrl = 'http://www.tucsonweekly.com/tucson/EventSearch?narrowByDate='
    
    # while loop to crawl over pages
    # Set maximum possible number of pages of events: 
    # in practice would be a huge number, and the code should break before this
    npageMax = 2 # for illustration, we'll stop after one page

    
    # base URL for this date
    url = baseUrl + dateString
    ii=0
    for pageNum in xrange(1, npageMax):
        print "On page =", pageNum
    
        # crawl over pages via appending initial date url
        if (pageNum>1):
            url = baseUrl + dateString + "&page=" + str(pageNum)
        
        # parse webpage with BeautifulSoup
        text = requests.get(url).text  # get raw html
        soup = BeautifulSoup(text)     # input it into BeautifulSoup to parse it
        
        # get all "div" tags that corresond to a single event
        eventTags = soup.find_all('div', 'EventListing clearfix')
        print "There are", len(eventTags), "events on this page"
        
        nEvents += len(eventTags)
        
{% endhighlight %}

    DATE:
    2016-02-07
    On page = 1
    There are 15 events on this page

 
So now we've found the events, we need to grab their data!

And this is where it got more irritating. Within each:

    <div class="EventListing clearfix" ...

I needed to find the url of the event, and unfortunately unlike the Village
Voice, Tucson Weekly wasn't nice in just allowing me to grab all anchor tags
with the attribute `href` like this:

    artistLinks = [tag.a.attrs['href'] for tag in artistTags]

Each event had multiple anchor tags, many with an `href` attribute that was not
the url I was after.

I managed to find the following fix: iterate over each anchor tag looking for
the first one that has a `href` attribute, but does NOT have a `class`
attribute. This seems to return the correct anchor tag with the `href` attribute
giving the event's url. 

**In [18]:**

{% highlight python %}
 
# loop over each event and attempt to extract the html link for the page of the event
for i, event in enumerate(eventTags):

    print "EVENT", i ,"START:"
        
    # find all the anchor ('a') tags within the 'div' tag for the evetn
    anchor_tags = event.find_all('a')
        
        
    # iterate over each anchor tag looking for the FIRST one that has a 'href' attribute
    # but does NOT have a 'class' attribute
    isFound = False
    for j,mb in enumerate(anchor_tags):
        
        if (mb.has_attr('href') and (not isFound) and (not mb.has_attr('class')) ):
            event_name = mb.get_text().strip() # this should be the event name! (type unicode)
            event_link = mb.attrs['href']      # this should be the event webpage! (type string)
            
            print "Event name =", event_name
            print "Event link =", event_link
            
            isFound = True
{% endhighlight %}

    EVENT 0 START:
    Event name = 43rd Annual President’s Concert
    Event link = http://www.tucsonweekly.com/tucson/43rd-annual-presidents-concert/Event?oid=6015139
    EVENT 1 START:
    Event name = Special Liturgical Music: Arvo Pärt's Berliner Messe
    Event link = http://www.tucsonweekly.com/tucson/special-liturgical-music-arvo-parts-berliner-messe/Event?oid=6022644
    EVENT 2 START:
    Event name = How to Live in Happiness
    Event link = http://www.tucsonweekly.com/tucson/how-to-live-in-happiness/Event?oid=6021947
    EVENT 3 START:
    Event name = Big Game Viewing Party
    Event link = http://www.tucsonweekly.com/tucson/big-game-viewing-party/Event?oid=6025673
    EVENT 4 START:
    Event name = Community Forum for LGBT Seniors and Friends
    Event link = http://www.tucsonweekly.com/tucson/community-forum-for-lgbt-seniors-and-friends/Event?oid=6021664
    EVENT 5 START:
    Event name = Reel in the Closet
    Event link = http://www.tucsonweekly.com/tucson/reel-in-the-closet/Event?oid=6020267
    EVENT 6 START:
    Event name = Your Wellness Journey-WellWays Workshop
    Event link = http://www.tucsonweekly.com/tucson/your-wellness-journey-wellways-workshop/Event?oid=6019458
    EVENT 7 START:
    Event name = Tucson Ukulele Meetup
    Event link = http://www.tucsonweekly.com/tucson/tucson-ukulele-meetup/Event?oid=6017963
    EVENT 8 START:
    Event name = Ron DeVous
    Event link = http://www.tucsonweekly.com/tucson/ron-devous/Event?oid=6022296
    EVENT 9 START:
    Event name = Ron Doering & RonDeVous Revue
    Event link = http://www.tucsonweekly.com/tucson/ron-doering-and-rondevous-revue/Event?oid=6014090
    EVENT 10 START:
    Event name = Art Walk Sundays
    Event link = http://www.tucsonweekly.com/tucson/art-walk-sundays/Event?oid=6008150
    EVENT 11 START:
    Event name = Art & Crafts Festival
    Event link = http://www.tucsonweekly.com/tucson/art-and-crafts-festival/Event?oid=6015383
    EVENT 12 START:
    Event name = Rhythms of the Americas
    Event link = http://www.tucsonweekly.com/tucson/rhythms-of-the-americas/Event?oid=5998815
    EVENT 13 START:
    Event name = Sorne - live performances & vocal workshop
    Event link = http://www.tucsonweekly.com/tucson/sorne-live-performances-and-vocal-workshop/Event?oid=6011617
    EVENT 14 START:
    Event name = "Desert Schemes"
    Event link = http://www.tucsonweekly.com/tucson/desert-schemes/Event?oid=5981354

 
Now all that is left is to go to each individual event url and scrape the key
information.

Following the blog post I define my own `scrape` function and modify to fit with
the Tucson Weekly event properties.

Again this is much tougher than the Village Voice case. Not all events have the
same properties, and even the ones that do have some variation between the
format of the data.

Also the `div` tags were not as straightforward, for example I couldn't just
call:

    find('div', 'when')
    find('div', 'price')
    find('div', 'neighborhood')

on the BeautifulSoup object, but instead look in the meta data and in a `div`
tag with `class="MainColumn Event"` and `id="EventMetaData"`

Then it gets even more fudgey where I have to use the `span` tag, which here
always has `class="label"` no matter the event property it contains. This means
I have to find the property type from the _actual text_ covered by the `span`. 

**In [20]:**

{% highlight python %}
import re

# function to scrape info from a given link on site
def scrape(link):

    # parse webpage with BeautifulSoup
    text = requests.get(link).text
    soup = BeautifulSoup(text)
    
    # extract venue and genre(s) from <title>stuff|stuff|stuff|</title>
    # at top of html
    title = soup.title.string
    title = title.split(' | ')
    name = title[0]
    venue = title[1]
    genre = title[2]
    print "Event name =", name
    print "Venue =", venue 
    print "Kind =", genre ,"\n"
    
    
    ### Get Description
    description = soup.find_all("meta", {"name":"description"} )[0]["content"]
    print "Description =", description ,"\n"
    
    
    ### Get Address stuff
    address = ["og:street-address", "og:latitude", "og:longitude", "og:locality",
               "og:region", "og:postal-code"]
    addr_stuff = []
    for ad in address:
        result = soup.find_all("meta", {"property": ad})
        if (len(result)>0):
            addr_stuff.append(result[0]["content"])
        else:
            addr_stuff.append(None)
        
    
    street = addr_stuff[0]
    lat = addr_stuff[1]
    lon = addr_stuff[2]
    locale = addr_stuff[3]
    region = addr_stuff[4]
    zipc = addr_stuff[5]
    
    print "Address =", street, locale, region, zipc
    print "Lat, lon =", lat, lon ,"\n"
    

    ### Get event meta data
    metaData = soup.find_all("div", {"class":"MainColumn Event ", "id":"EventMetaData"} )
    spans = metaData[0].find_all("span")
    
    ### Look for text indicating event property within each span tag
    foundWhen = False
    foundPrice = False
    textPrice = 'none'
    for i, cont in enumerate(spans):

        if (cont.text=="When:"):
            when = cont.next_sibling.strip()
            foundWhen = True
            
        if (cont.text=="Price:"):
            textPrice = cont.next_sibling.strip()
            foundPrice = True
            

            
    ### Some events don't have a 'when' property, 
    #   e.g. they just say 'every 4th Friday within event description
    if (not foundWhen):
        print "returning none"
        return None
            

    ### Parse price value
    
    # use regular expressions (re) to scrape multiple prices
    # ()    look for groups of the character sequence inside the brackets
    # \$    look for a dollar sign at the front (needs escape character \)
    # \d    look for decimal digit [0-9]
    # +     greedy, make longest sequence possible (match one or more of the preciding RE)
    # \.    look for '.' (needs escape character \)
    # ?     Causes the resulting RE to match 0 or 1 repetitions of the preceding RE. 
    #       ab? will match either 'a' or 'ab'. i.e. doesn't HAVE to match the thing immediately proceeding it
    
    dollarValues = re.findall('(\$?\d+\.?\d?\d?)', textPrice)
    
    # checks if price contains free 
    if (textPrice.count('free') != 0) or (textPrice.count('Free') != 0) or (textPrice.count('FREE') != 0): 
        minprice = maxprice = 0.00
    elif len(dollarValues) == 0:
        minprice = maxprice = None
    else:
        for i in range(0, len(dollarValues)):
            dollarValues[i] = float(dollarValues[i].strip('$'))
        minprice = min(dollarValues)
        maxprice = max(dollarValues)

        
    if (maxprice != minprice):
        print "Price range =", minprice, "to", maxprice
        price = str(minprice) + " to " + str(maxprice)
    else:
        price = minprice
        print "Price =", minprice
    
    print "\nWhen =", when ,"\n"
    
    return name, description, street + locale + region + zipc, price, when

    
# returns tuple of name, description, street + locale + region + zipc, price, when
print scrape(event_link)
{% endhighlight %}

    Event name = "Desert Schemes"
    Venue = Desert Artisans' Gallery
    Kind = Art 
    
    Description = “Desert Schemes” new art exhibit through Feb 7th works by Margaret Aden, Gail Brynildsen, Denyse Fenelon, Pamela Howe, Tom Kolt and Jan Thompson. 
    
    Address = 6536 E. Tanque Verde Road. Tucson Arizona 
    Lat, lon = 32.24576 -110.85340 
    
    Price = 0.0
    
    When = First Monday-Sunday of every month, 10 a.m.-5 p.m. Continues through Feb. 7 
    
    (u'"Desert Schemes"', u'\u201cDesert Schemes\u201d new art exhibit through Feb 7th works by Margaret Aden, Gail Brynildsen, Denyse Fenelon, Pamela Howe, Tom Kolt and Jan Thompson.', '6536 E. Tanque Verde Road.TucsonArizona', 0.0, u'First Monday-Sunday of every month, 10 a.m.-5 p.m. Continues through Feb. 7')

 
In short, we won!

In long, there are still a bunch of problems:

  1. the returned data is not in a consistent format
  2. if the event has no "When: " text it is not included, even though the
"when" is probably given in the event description 


{% highlight python %}

{% endhighlight %}
