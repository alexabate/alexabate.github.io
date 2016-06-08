---
layout: post
title: "map-income-tucson"
tags:
    - python
    - notebook
--- 
# Hack for Change mapping project

4th June 2016 was [Code For America's "National Day of Civic
Hacking"](https://cache.codeforamerica.org/events/national-day-2016/) (or "Hack
for Change" which is more speak-able). Tucson's local event was held in the
University of Arizona's Science and Engineering library. The temperature in
Tucson had just popped over 100F, that day in particular was forecast to be
113F, so there never was a better day for staying indoors hacking with plenty of
air conditioning (and pizza).

Looking through the list of suggested projects I found the [Opportunity
Project](https://cache.codeforamerica.org/events/national-day-2016/challenge-
the-opportunity-project) particularly interesting because it involved taking
advantage of federal+local open data for social good. It also gave the chance to
investigate the [CitySDK](https://uscensusbureau.github.io/citysdk/guides.html)
tool that works as a wrapper around the various APIs required to grab the
different data sets available (census, FEMA, farmer's markets, etc).

We formed at team of 3 (Jon Eckel, Pete Lowe and myself) called JustMapIt!
(chosen to reflect our dedication to producing *something* by the end of the
day). We decided that creating a visualisation that mapped the population income
and/or poverty index across Tucson, along with access to grocery stores, may
yield something interesting and useful. First we began by checking out the
available data, making sure it contained data in the Tucson area!

The first major issue was that the CitySDK tool didn't appear to be working. In
the interest of time we decided to directly grab our own data sets instead.


## Data sets

* INCOME IN THE PAST 12 MONTHS (IN 2014 INFLATION-ADJUSTED DOLLARS) from the
[2014 American Community Survey 1-Year Estimates](http://factfinder.census.gov/)
data for Arizona
* Latitude and Longitude positions of grocery stores in Tucson scraped from
venues with categoryID='Grocery Store' in Foursquare using its API (and then
cleaned a little)
 

**In [40]:**

{% highlight python %}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

%matplotlib inline
{% endhighlight %}
 
## The income data

This data set gives the number of households and median income per census tract.
Census tracts are small-ish, subdivisions of a county (or similar). Theys
provide a stable set of geographic units for the presentation of statistical
census data. Generally they contain a population size between 1,200 and 8,000
people, with an optimum size of 4,000 people.

In the data below the columns `id` and `id2` contain the census tract id's. 

**In [41]:**

{% highlight python %}
# household income census data
income_df = pd.read_csv("income_census_data.csv", header=[0,1], dtype={0:str, 1:str})

{% endhighlight %}
 
## Munging

There were two levels of column labels so the dataframe columns were
multindexed. Since the upper level of labels gave no useful information, for
ease of use we removed them. 

**In [42]:**

{% highlight python %}
# remove secondary column label
levels = income_df.columns.levels
labels = income_df.columns.labels
income_df.columns = levels[1][labels[1]]

# quick look at data
income_df.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Id2</th>
      <th>Geography</th>
      <th>Households; Estimate; Total</th>
      <th>Households; Margin of Error; Total</th>
      <th>Families; Estimate; Total</th>
      <th>Families; Margin of Error; Total</th>
      <th>Married-couple families; Estimate; Total</th>
      <th>Married-couple families; Margin of Error; Total</th>
      <th>Nonfamily households; Estimate; Total</th>
      <th>...</th>
      <th>Nonfamily households; Estimate; PERCENT IMPUTED - Family income in the past 12 months</th>
      <th>Nonfamily households; Margin of Error; PERCENT IMPUTED - Family income in the past 12 months</th>
      <th>Households; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Households; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Families; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Families; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Married-couple families; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Married-couple families; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Nonfamily households; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
      <th>Nonfamily households; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1400000US04019000100</td>
      <td>04019000100</td>
      <td>Census Tract 1, Pima County, Arizona</td>
      <td>319</td>
      <td>50</td>
      <td>48</td>
      <td>38</td>
      <td>34</td>
      <td>31</td>
      <td>271</td>
      <td>...</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>13.7</td>
      <td>(X)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1400000US04019000200</td>
      <td>04019000200</td>
      <td>Census Tract 2, Pima County, Arizona</td>
      <td>1916</td>
      <td>189</td>
      <td>914</td>
      <td>182</td>
      <td>452</td>
      <td>145</td>
      <td>1002</td>
      <td>...</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>26.7</td>
      <td>(X)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1400000US04019000300</td>
      <td>04019000300</td>
      <td>Census Tract 3, Pima County, Arizona</td>
      <td>680</td>
      <td>86</td>
      <td>244</td>
      <td>54</td>
      <td>109</td>
      <td>54</td>
      <td>436</td>
      <td>...</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>22.5</td>
      <td>(X)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1400000US04019000400</td>
      <td>04019000400</td>
      <td>Census Tract 4, Pima County, Arizona</td>
      <td>1719</td>
      <td>97</td>
      <td>395</td>
      <td>101</td>
      <td>253</td>
      <td>78</td>
      <td>1324</td>
      <td>...</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>27.5</td>
      <td>(X)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1400000US04019000500</td>
      <td>04019000500</td>
      <td>Census Tract 5, Pima County, Arizona</td>
      <td>1544</td>
      <td>119</td>
      <td>309</td>
      <td>98</td>
      <td>158</td>
      <td>64</td>
      <td>1235</td>
      <td>...</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>(X)</td>
      <td>30.8</td>
      <td>(X)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 131 columns</p>
</div>



**In [43]:**

{% highlight python %}
print "Name of columns with income data:"
for col in income_df.columns:
    if "income" in col:
        print col

{% endhighlight %}

    Name of columns with income data:
    Households; Estimate; Median income (dollars)
    Households; Margin of Error; Median income (dollars)
    Families; Estimate; Median income (dollars)
    Families; Margin of Error; Median income (dollars)
    Married-couple families; Estimate; Median income (dollars)
    Married-couple families; Margin of Error; Median income (dollars)
    Nonfamily households; Estimate; Median income (dollars)
    Nonfamily households; Margin of Error; Median income (dollars)
    Households; Estimate; Mean income (dollars)
    Households; Margin of Error; Mean income (dollars)
    Families; Estimate; Mean income (dollars)
    Families; Margin of Error; Mean income (dollars)
    Married-couple families; Estimate; Mean income (dollars)
    Married-couple families; Margin of Error; Mean income (dollars)
    Nonfamily households; Estimate; Mean income (dollars)
    Nonfamily households; Margin of Error; Mean income (dollars)
    Households; Estimate; PERCENT IMPUTED - Household income in the past 12 months
    Households; Margin of Error; PERCENT IMPUTED - Household income in the past 12 months
    Families; Estimate; PERCENT IMPUTED - Household income in the past 12 months
    Families; Margin of Error; PERCENT IMPUTED - Household income in the past 12 months
    Married-couple families; Estimate; PERCENT IMPUTED - Household income in the past 12 months
    Married-couple families; Margin of Error; PERCENT IMPUTED - Household income in the past 12 months
    Nonfamily households; Estimate; PERCENT IMPUTED - Household income in the past 12 months
    Nonfamily households; Margin of Error; PERCENT IMPUTED - Household income in the past 12 months
    Households; Estimate; PERCENT IMPUTED - Family income in the past 12 months
    Households; Margin of Error; PERCENT IMPUTED - Family income in the past 12 months
    Families; Estimate; PERCENT IMPUTED - Family income in the past 12 months
    Families; Margin of Error; PERCENT IMPUTED - Family income in the past 12 months
    Married-couple families; Estimate; PERCENT IMPUTED - Family income in the past 12 months
    Married-couple families; Margin of Error; PERCENT IMPUTED - Family income in the past 12 months
    Nonfamily households; Estimate; PERCENT IMPUTED - Family income in the past 12 months
    Nonfamily households; Margin of Error; PERCENT IMPUTED - Family income in the past 12 months
    Households; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Households; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Families; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Families; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Married-couple families; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Married-couple families; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Nonfamily households; Estimate; PERCENT IMPUTED - Nonfamily income in the past 12 months
    Nonfamily households; Margin of Error; PERCENT IMPUTED - Nonfamily income in the past 12 months

 
It seems the column we want to look at is "Households; Estimate; Median income
(dollars)" 

**In [44]:**

{% highlight python %}
median_income = income_df["Households; Estimate; Median income (dollars)"]
print median_income.describe()

print median_income[:30]


#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.hist(median_income)

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.hist(median_income)

{% endhighlight %}

    count       241
    unique      240
    top       27472
    freq          2
    Name: Households; Estimate; Median income (dollars), dtype: object
    0     24861
    1     24856
    2     30739
    3     18792
    4     23188
    5     51667
    6     25805
    7     44250
    8     34492
    9     39145
    10    26983
    11    30441
    12    13193
    13    22955
    14    14940
    15    21599
    16    27573
    17    50387
    18    41507
    19    28874
    20    33947
    21    48258
    22    33380
    23    28247
    24    32857
    25    28084
    26    23778
    27    24214
    28    29292
    29    30878
    Name: Households; Estimate; Median income (dollars), dtype: object

 
The output from `describe` looks odd though the data itself looks ok. Also an
error is raised on trying to plot it, with `hist`

    TypeError: len() of unsized object

or with `plot`

    ValueError: could not convert string to float:

 

**In [45]:**

{% highlight python %}
# check by eye
#for val in median_income:
#    print val, type(val)

    
# Sample output:
# 84410 <type 'str'>
# 65284 <type 'str'>
# 53460 <type 'str'>
# 39798 <type 'str'>
# - <type 'str'>
# 29923 <type 'str'>
# 27664 <type 'str'>
# 34726 <type 'str'>
# 34000 <type 'str'>

# print all entries with "-" for Households; Estimate; Median income (dollars)
for index, row in income_df.iterrows():
    if row["Households; Estimate; Median income (dollars)"]=='-':
        print "Median income =", row["Households; Estimate; Median income (dollars)"],
        print "Number of households =", row["Households; Estimate; Total"]
{% endhighlight %}

    Median income = - Number of households = 0

 
There is one entry where there is a null value (`-`) for the median income, and
this corresponds to a census tract with 0 households (this seems to be because
this tract is a State Prison complex).
So we need to ignore the tract where number of households=0, and also convert
the data to floats (because its type is string).
 

**In [46]:**

{% highlight python %}
median_income = income_df.ix[(income_df["Households; Estimate; Total"] > 0), 
                             "Households; Estimate; Median income (dollars)"].astype(float)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.hist(median_income, 50)
ax.set_xlabel('median income in AZ per tract ($)', fontsize=24)


{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/map-income-tucson_12_1.png) 

 
## Read in the Tucson grocery store data

Scraped from foursquare into a simple CSV file 

**In [47]:**

{% highlight python %}
supermarkets = pd.read_csv("grocery_stores.csv")
supermarkets.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lon</th>
      <th>name</th>
      <th>addr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.229253</td>
      <td>-110.873651</td>
      <td>Kimpo Market</td>
      <td>5595 E 5th St</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32.220195</td>
      <td>-110.807966</td>
      <td>Walmart Neighborhood Market</td>
      <td>8640 E Broadway Blvd</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.118384</td>
      <td>-110.798278</td>
      <td>Safeway</td>
      <td>9050 E Valencia Rd</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32.256930</td>
      <td>-110.943687</td>
      <td>India Dukaan</td>
      <td>2754 N Campbell Ave</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32.193137</td>
      <td>-110.841855</td>
      <td>Walmart Neighborhood Market</td>
      <td>2550 S Kolb Rd</td>
    </tr>
  </tbody>
</table>
</div>


 
## Folium for mapping

Folium is a python wrapper for the Leaflet javascript library, which itself can
render interactive maps.

We need a way to convert the census tract ID to its equivalent area on the map,
the census website provides this [data](https://www.census.gov/geo/maps-
data/data/cbf/cbf_tracts.html) in the form of ESRI Shapefiles.

Folium works with GeoJSON files so we need to convert. Handily we can do this
using an [online converter](http://ogre.adc4gis.com/) 

**In [48]:**

{% highlight python %}
import folium
import json

# GeoJSON file of Arizona census tracts
state_geo = "arizona.json"

# initialize map
tucson_coords = [32.2,-110.94]
mp = folium.Map(location=tucson_coords, zoom_start=11)

# map data to geo_json
mp.geo_json(geo_path=state_geo, 
            data=income_df.ix[(income_df["Households; Estimate; Total"] > 0)],
            data_out="median_income.json", 
            columns=["Id2", "Households; Estimate; Median income (dollars)"],
            key_on="feature.properties.GEOID",
            fill_color='YlGn',
            fill_opacity=0.7,
            line_opacity=0.2 ,
            threshold_scale= np.logspace(np.log10(15000), np.log10(125000), 6).tolist(),
            legend_name='Median Income') 

# plot the supermarkets on the map
for i,row in supermarkets.iterrows():
    name = row["name"].decode("utf8")
    mp.circle_marker(location=[str(row["lat"]), str(row["lon"])], popup=name, radius=100, fill_color="red", )

# generate the HTML/Javascript
mp.create_map(path='tucson.html', plugin_data_out=False)
{% endhighlight %}
 
## Map!

Here is the [map](http://u.arizona.edu/~abate/arizona.html). Unfortunately we
ran out of time before being able to add a toggle to toggle between median
income and another dataset (e.g. population density). This particular
visualisation would also be served better by higher resolution income data than
that given by census tracts, but it was a great start: we learned a lot and
finished *something*!

 




