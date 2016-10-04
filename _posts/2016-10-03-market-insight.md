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

To draw out the most important features of each restaurant I calculated the term frequency-inverse document frequency (TF-IDF) statistic on the corpus of restaurant comments. I chose TF-IDF because the final statistical weight of each term includes the inverse document frequency. This is important so terms that appear many times in only a few documents are still weighted highly (e.g. a popular but generally uncommon food item that is a speciality in a certain small subset of restaurants). The "terms" in the resulting restaurant - term matrix needed to be cleaned: numbers and words in cyrillic were removed and singular/plural pairs of the same word were aggregated.

