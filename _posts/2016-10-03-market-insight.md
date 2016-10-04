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

To draw out the most important features of each restaurant I calculated the term frequency-inverse document frequency (TF-IDF) statistic on the corpus of restaurant comments. I chose TF-IDF because the final statistical weight of each term includes the inverse document frequency. This is important so terms that appear many times in only a few documents are still weighted highly (e.g. a popular but generally uncommon food item that is a speciality in a certain small subset of restaurants). I chose to include uni-grams (single word) and bi-grams (two words together).

The "terms" in the resulting restaurant - term matrix needed to be cleaned: terms which were numbers or words in cyrillic were removed via regular expression search, and singular/plural pairs of the same word were aggregated.

I then clustered the restaurants based upon their term weight values. The idea of this is to refine the extremely noisy categorisation of the restaurants by Foursquare users, down from over 200 separate categories (ones that included "Museum" and "Pet Store" to a more manageable number. I used k means to find 20 clusters based upon a distance between restaurants in TF-IDF term-weight space defined by the cosine similarity. I chose k means to begin with because of its simplicity and also because I had a good idea of what the number of k clusters should be a priori. Manually assigning the 200+ Foursquare categories into restaurants with different cuisine types (or into "other" type) resulted in about 20 reasonably broad categories. 

I validated the clusters (and the choice of k=20) by studying the frequencies of the top Foursquare categories, the top weighted terms, and the price distribution of all restaurants in the cluster. The clusters came out as shown in the following list. The head of the list gives the most frequent Foursquare category in the cluster, the five words after the colon are the top five terms in that cluster.

* Mexican Restaurant: taco, margarita, burrito, mexican, guacamole
* Ramen Restaurant (Noodles): ramen, broth, pork, noodle, spicy
* American Restaurant (Brunch and Breakfast): brunch, egg, burger, cocktail, breakfast
* Pizza place (Pizza cheap): pizza, slice, best pizza, crust, good pizza
* Burger Joint: burger, fries, shake , best burger, shack
* Pizza Place (Pizza expensive): pizza, slice, pasta, pepperoni, crust
* Deli/Bodega (other): rice, sandwich, falafel, line, bowl
* French Restaurant (Fancy): steak, wine, cocktail, brunch, lamb, oyster
* Coffee Shop (Coffee general): latte, iced, espresso, wifi, brew
* Bagel Shop: bagel, cream cheese, cream, lox, egg 
* Sandwich Place: sandwich, soup, breakfast, egg, wrap
* Chinese Restaurant (Asian): noodle, dumpling, thai, soup, pork
* Deli/Bodega (other): donut, deli, harlem, guy, sandwiches
* Sushi Restaurant: sushi, roll, sashimi, lunch special, tuna
* Italian Restaurant: pasta, italian, wine, gnocchi, ravioli
* Bar: beer, happy hour, hour, happy, bartender
* Ice Cream Shop (Bakery): cookie, chocolate, ice cream, cream, ice
* Juice Bar (Juice/Veg/Healthy): juice, smoothie, green, bowl, protein
* Coffee Shop (Starbucks): starbucks, barista, line, bathroom, outlet
* Bubble Tea Shop (Tea and Boba): tea, bubble, bubble tea, milk, green tea

In most cases the most frequent Foursquare category is assigned to be the label of that cluster. However, except when there is a word in brackets after it, this is assigned to be the label instead. A couple of notes about the labelling: the "pizza" labels were assigned by looking at the price differences between the two clusters: one was clearly skewed low and the other high. The "Starbucks" label was assigned because 99% of all the restaurants in this cluster were Starbucks, while zero of the restaurants in the other coffee category were Starbucks.

