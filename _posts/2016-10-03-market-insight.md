---
layout: post
title: "Market Insight"
tags:
    - python
    - notebook
--- 
## Market Insight: Visualise the potential of NYC restaurant markets

My project was to use [Foursquare](https://foursquare.com/) restaurant comments data and [Census demographics data](http://www.census.gov/data.html) to develop an analytics tool that shows the features of a neighborhood so that a business could decide how/if to enter into the restaurant market there.
 
My approach to this problem, to identify the potential customers and their interests in order to enable an estimation of the market size, is to combine crowdsourced and public data sets of all Manhattan neighborhoods.
 
# Data collection and cleaning

For the "crowdsourced" data I chose the user comments (called "tips") left for restaurants on Foursquare. These comments provide a text corpus to mine to learn the specialities and categories of particular restaurants. They also provide metadata for the restaurant, including its location, a Foursquare-defined category and a price tier (between one and four $-signs).
 
The Foursquare API allows you to search for venues within a specified radius around a latitude, longitude point and will return up to a max of 50 venues for a single call. I specified the venue category to be the category ID corresponding to "Food" and performed a few thousand calls to the API over a grid of latitude, longitude defined by the locations of Census tracts in Manhattan. The number of Census tracts in Manhattan is 288 so I interpolated extra latitude, longitude points between these values to create grid. Because the API only allows 2500 calls per hour, I added in a condition for the code to pause so as not to exceed the API limit. I gathered data for around 10,500 restaurants.

To create the text corpus of restaurant comments I concatentated all the comment strings for each restaurant into a single long string: that became the "document" for a restaurant. Any restaurants with zero comments were discarded.

To draw out the most important features of each restaurant I calculated the term frequency-inverse document frequency (TF-IDF) statistic on the corpus of restaurant comments. I chose TF-IDF because the final statistical weight of each term includes the inverse document frequency. This is important so terms that appear many times in only a few documents are still weighted highly (e.g. a popular but generally uncommon food item that is a speciality in a certain small subset of restaurants). The "terms" in the resulting restaurant - term matrix needed to be cleaned: terms which were numbers or words in cyrillic were removed via regular expression search, and singular/plural pairs of the same word were aggregated.

I then clustered the restaurants based upon their term weight values. The idea of this is to refine the extremely noisy categorisation of the restaurants by Foursquare users, down from over 200 separate categories (ones that included "Museum" and "Pet Store" to a more manageable number. I used k means to find 20 clusters based upon a distance between restaurants in TF-IDF term-weight space defined by the cosine similarity. I chose k means to begin with because of its simplicity and also because I had a good idea of what the number of k clusters should be a priori. Manually assigning the 200+ Foursquare categories into restaurants with different cuisine types (or into "other" type) resulted in about 20 reasonably broad categories. 

I validated the clusters (and the choice of k=20) by studying the frequencies of the top Foursquare categories, the top weighted terms, and the price distribution of all restaurants in the cluster.


| Mexican Restaurant          | Ramen Restaurant | American Restaurant | Pizza Place | Burger Joint |
| taco            106.074854  |
| margarita        56.581771  | 
| burrito          50.611620  |
| mexican          33.425526  |
| guacamole        26.936074  |
| guac             18.633560  |
| chipotle         17.403008  |
| quesadilla       16.030553  |
| salsa            14.033628  |
| mexican food     13.589426  |


ramen        46.509661
broth         7.054914
pork          6.607530
noodle        6.235213
spicy         6.001931
miso          5.152043
bun           5.108965
pork buns     3.396329
gyoza         2.841215
japanese      2.072981

dtype: float64

American
Restaurant
brunch       58.564862
egg          35.626095
burger       35.155186
cocktail     26.592421
breakfast    26.360047
diner        25.258278
french       22.197736
pancake      21.463557
toast        19.448069
fries        19.387919
dtype: float64

Pizza
Place
pizza          168.125799
slice           76.093101
best pizza      19.976093
crust           16.249763
good pizza      11.896737
pie             11.343058
great pizza      9.654576
pepperoni        9.112524
pizza great      8.724700
grandma          8.473557
dtype: float64

Burger
Joint
burger          113.913363
fries            29.108766
shake            15.948698
best burger      11.858803
shack             9.494405
beer              9.406615
milkshake         8.208241
onion             8.162103
great burger      7.965306
good burger       6.742418
dtype: float64

Pizza
Place
pizza          66.915124
slice          20.785261
pasta           8.877717
pepperoni       6.231597
crust           6.163282
best pizza      4.757829
italian         4.548118
pie             4.520934
good pizza      4.415230
great pizza     3.940523
dtype: float64

Deli
/
Bodega
rice         35.103138
sandwich     30.878433
falafel      29.878026
line         23.698764
bowl         23.630598
breakfast    21.508876
egg          21.191393
option       20.082750
fries        20.066916
cheap        19.868196
dtype: float64

French
Restaurant
steak       40.455698
wine        39.416677
cocktail    34.621441
brunch      24.530587
lamb        21.163313
oyster      20.124585
fried       20.031611
dessert     19.898883
hour        19.405938
lobster     18.527887
dtype: float64

Coffee
Shop
latte           54.180277
iced            32.881011
espresso        32.602134
wifi            28.775551
brew            26.508900
shop            24.054733
coffee shop     23.082525
cold brew       22.813944
great coffee    21.999532
cappuccino      19.790833
dtype: float64

Bagel
Shop
bagel           78.288918
cream cheese    17.014826
cream           11.154938
lox              9.166384
egg              5.300773
toasted          4.384175
breakfast        3.847934
sandwich         3.585684
sandwiches       3.480765
wheat            3.375339
dtype: float64

Sandwich
Place
sandwich      75.541950
sandwiches    56.893923
soup          45.043977
breakfast     42.211346
egg           28.233152
wrap          25.435358
salads        24.528797
bread         19.665863
deli          19.462199
line          19.090226
dtype: float64

Chinese
Restaurant
noodle      61.192954
dumpling    53.544949
thai        52.106745
soup        33.170717
pork        31.224108
pad         28.558166
rice        27.332594
spicy       24.176786
curry       22.446037
fried       21.904150
dtype: float64

Deli
/
Bodega
donut         19.923673
deli          16.734727
harlem        13.502923
guy           12.453738
sandwiches    12.023307
sandwich      11.319443
chinese       11.269767
dog           10.996743
prices        10.539961
bodega        10.422657
dtype: float64

Sushi
Restaurant
sushi            97.805848
roll             68.743827
sashimi          14.771224
lunch special    14.096545
tuna             13.022619
spicy            12.733249
best sushi       11.565321
tempura          10.447925
japanese         10.162907
salmon            9.653704
dtype: float64

Italian
Restaurant
pasta           56.849868
italian         38.853512
wine            33.954099
gnocchi         13.452074
ravioli         13.161733
spaghetti       12.576442
meatball        11.522954
italian food    10.720165
pizza            9.703508
veal             9.691566
dtype: float64

Bar
beer          69.537537
happy hour    32.069424
hour          29.331885
happy         27.681802
bartender     26.168990
wing          24.813113
burger        23.703924
selection     17.775067
game          16.774174
night         16.002872
dtype: float64

Ice
Cream
Shop
cookie       39.634677
chocolate    36.887078
ice cream    33.941215
cream        33.584652
ice          31.216984
cupcake      25.110845
flavor       18.803872
cake         18.598754
chip         11.825379
gelato       11.223432
dtype: float64

Juice
Bar
juice       41.964850
smoothie    33.911946
green       11.616078
bowl        11.087107
protein      5.931110
healthy      5.421453
kale         4.799185
almond       4.601314
butter       4.198994
vegan        4.162210
dtype: float64

Coffee
Shop
starbucks    78.158267
barista      25.907918
line         18.353533
bathroom     14.538820
outlet       14.460722
morning      13.601044
location     12.638338
latte        12.346729
slow         12.014702
wifi          9.902656
dtype: float64

Bubble
Tea
Shop
tea           50.961252
bubble        23.128059
bubble tea    20.533554
milk          11.608776
green tea      5.969694
green          5.740394
scone          4.530234
sugar          3.913762
iced           3.547090
jelly          3.043388
