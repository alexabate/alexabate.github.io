---
layout: post
title: "Adding data from Foursquare"
tags:
    - python
    - notebook
--- 
# Adding data from Foursquare

The Tucson Weekly event listings contain data that includes the venue name,
address and latitude and longitude. A way to add value to the listings search
would be to augment the database with more properties of each venue.

Below I implemented a function that grabs all venues from the Foursquare
database (well up to 50 is what's allowed) within 300m of a supplied latitude
and longitude.

The function then finds close matches the name that is supplied, if that fails
it can use the supplied address. The closest venue with a closely matching name
(or address) is taken to be a correct "match".

The function returns some useful information to add to the database: a bitmap
that describes what "services" the venue provides, a bitmap that describes the
venue's opening hours, and the rating, the number of likes and the venue 'tags'
from foursquare.

(The bitmaps are effectively just numbers in binary and will be converted to
decimal integers for storage in the database) 

**In [3]:**

{% highlight python %}


### Here is some venue information scraped from the Tucson Weekly
venues = [(32.22210, -110.96693, 'Club Congress'), 
          (32.20857, -110.91799, 'Reid Park Zoo'),
          (32.20630, -110.90033, 'Eckstrom-Columbus Branch Library'),
          (32.23302, -110.95744, 'Arizona History Museum'),
          (32.26066, -110.98067, 'Monterey Court'),
          (31.85535, -111.00729, 'CPAC Community Performance & Art Center'),
          (32.22190, -110.97168, 'Fox Tucson Theatre'),
          (32.27204, -110.98700, 'Loudhouse Rock and Roll Bar and Grill'),
          (32.23626, -110.92357, 'Loft Cinema'),
          (32.29105, -110.99575, 'The EDGE', '4635 N Flowing Wells Rd')] 

{% endhighlight %}

**In [4]:**

{% highlight python %}
import utes

### Test how these venues are found within foursquare
radius = 300
for venue in venues:

    lat = venue[0]
    lon = venue[1]
    name = venue[2]
    
    if len(venue)>3:
        result = utes.getVenue(lat, lon, name, radius, venue[3]) 
    else:
        result = utes.getVenue(lat, lon, name)
        
    print "RETURNED RESULTS FOR", name ,":"
    if len(result)>1:
        print "services: [beer, coffee, food, liquor, wine] =", result[0]
        print "hours:", result[1]
        print "rating:", result[2]
        print "number of likes:", result[3]
        print "venue type:", result[4]
    else:
        print "Failed to find venue"
    print "\n\n\n"
{% endhighlight %}

    Returned 45 venues within 300 meters
    Name given = Club Congress
    Name in foursquare = Club Congress
    Distance from original lat, long = 36 meters
    No hours in venue information
    
    RETURNED RESULTS FOR Club Congress :
    services: [beer, coffee, food, liquor, wine] = [1, 0, 0, 1, 1]
    hours: None
    rating: 8.6
    number of likes: 116
    venue type: [u'Rock Club', u'Karaoke Bar', u'Bar']
    
    
    
    
    Returned 29 venues within 300 meters
    Name given = Reid Park Zoo
    Name in foursquare = Reid Park Zoo
    Distance from original lat, long = 272 meters
    No hours in venue information
    
    RETURNED RESULTS FOR Reid Park Zoo :
    services: [beer, coffee, food, liquor, wine] = [0, 0, 1, 0, 0]
    hours: None
    rating: 8.7
    number of likes: 139
    venue type: [u'Zoo']
    
    
    
    
    Returned 34 venues within 300 meters
    Name given = Eckstrom-Columbus Branch Library
    Name in foursquare = Eckstrom Columbus Library
    Distance from original lat, long = 115 meters
    No hours in venue information
    No rating in venue information
    
    RETURNED RESULTS FOR Eckstrom-Columbus Branch Library :
    services: [beer, coffee, food, liquor, wine] = [0, 0, 0, 0, 0]
    hours: None
    rating: None
    number of likes: 0
    venue type: [u'College Library']
    
    
    
    
    Returned 39 venues within 300 meters
    RETURNED RESULTS FOR Arizona History Museum :
    Failed to find venue
    
    
    
    
    Returned 20 venues within 300 meters
    Name given = Monterey Court
    Name in foursquare = Monterey Court
    Distance from original lat, long = 13 meters
    
    RETURNED RESULTS FOR Monterey Court :
    services: [beer, coffee, food, liquor, wine] = [0, 0, 1, 0, 0]
    hours: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
       1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
       1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
       1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
       1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
       1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    rating: 5.3
    number of likes: 4
    venue type: [u'American Restaurant']
    
    
    
    
    Returned 1 venues within 300 meters
    RETURNED RESULTS FOR CPAC Community Performance & Art Center :
    Failed to find venue
    
    
    
    
    Returned 26 venues within 300 meters
    Name given = Fox Tucson Theatre
    Name in foursquare = Fox Tucson Theatre
    Distance from original lat, long = 21 meters
    No hours in venue information
    
    RETURNED RESULTS FOR Fox Tucson Theatre :
    services: [beer, coffee, food, liquor, wine] = [0, 0, 0, 0, 0]
    hours: None
    rating: 7.8
    number of likes: 30
    venue type: [u'Music Venue', u'Indie Movie Theater', u'Monument / Landmark']
    
    
    
    
    Returned 42 venues within 300 meters
    Name given = Loudhouse Rock and Roll Bar and Grill
    Name in foursquare = Loudhouse Rock and Roll Bar and Grill
    Distance from original lat, long = 11 meters
    No hours in venue information
    No rating in venue information
    
    RETURNED RESULTS FOR Loudhouse Rock and Roll Bar and Grill :
    services: [beer, coffee, food, liquor, wine] = [1, 0, 0, 0, 1]
    hours: None
    rating: None
    number of likes: 0
    venue type: [u'Bar']
    
    
    
    
    Returned 50 venues within 300 meters
    Name given = Loft Cinema
    Name in foursquare = The Loft Cinema
    Distance from original lat, long = 20 meters
    No hours in venue information
    
    RETURNED RESULTS FOR Loft Cinema :
    services: [beer, coffee, food, liquor, wine] = [1, 0, 1, 0, 0]
    hours: None
    rating: 9.1
    number of likes: 93
    venue type: [u'Movie Theater', u'Indie Movie Theater']
    
    
    
    
    Returned 8 venues within 300 meters
    Name given = The EDGE
    Name in foursquare = River's Edge Lounge
    Distance from original lat, long = 4 meters
    
    RETURNED RESULTS FOR The EDGE :
    services: [beer, coffee, food, liquor, wine] = [1, 0, 0, 1, 1]
    hours: [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
       0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    rating: 7.3
    number of likes: 8
    venue type: [u'Karaoke Bar']
    
    
    
    

 
## Successes

The code does seem to be able to find the correct venues in foursquare, even
when they are not the closest distance away (as compared to foursquare's
latitude and longitude) and when there are small perturbations in the names,
e.g. "Loft Cinema" vs "The Loft Cinema" and "Eckstrom-Columbus Branch Library"
vs "Eckstrom Columbus Library"

It's even potentially robust to large differences in the recorded name, e.g.
"The EDGE" vs "River's Edge Lounge", because the code can resort to comparing
street addresses if no close name match is found.

## Issues
* Not many of the `service_flags` correctly represent everything a venue offers,
e.g. The Loft Cinema also serves coffee and wine but only beer and food are
recorded
* Foursquare is not abundantly used by the Tucson community so there's a lot of
missing information, not many 'tips' to scrape information from
* As above, very few of the venues actually contain any 'hours' information
* Incorrect data: the hours in foursquare listed for the River's Edge Lounge are
11am to 2pm when clearly it's supposed to be 2am
* On the foursquare venue's page there is often more useful information I
haven't figured out how to access through the API yet, e.g.
    - `Menus: Brunch, Lunch, Dinner`
    - `Drinks: Wine`
    - `Credit Cards: Yes`
    - `Outdoor Seating: No`
* If a venue is not found I can't tell if it's because the venue is not in
foursquare or the algorithm failed.
* Much more sophisticated work using NLP will need to be done to make the most
use of all the information in the foursquare venue (mostly from 'tips' left by
users)
 
