---
layout: post
title: "Market Insight"
tags:
    - python
    - notebook
--- 
## Market Insight: Visualise the potential of NYC restaurant markets

I created the **Market Insight** tool over a period of three weeks during the [_Insight Data Science_](www.insightdatascience.com) program.

The project used [_Foursquare_](https://foursquare.com/) restaurant comments data and [_Census demographics data_](http://www.census.gov/data.html) to develop an analytics tool that shows the features of a neighborhood so that a business can gain actionable insights into the restaurant market.
 
My approach to this problem, to identify the potential customers and their interests in order to enable an estimation of the market size, is to combine crowdsourced and public data sets of all Manhattan neighborhoods. 

A link to the presentation slides can be found [_here_](http://www.slideshare.net/secret/ADOxPCif1zZdq4).
 
# Data collection and cleaning

For the "crowdsourced" data I chose the user comments (called "tips") left for restaurants on Foursquare. These comments provide a text corpus to mine to learn the specialities and "hot topics" associated with particular restaurants. They also provide metadata for the restaurant, including its location, a Foursquare-defined category and a price tier (between one and four $-signs).
 
 <img align="right" src="{{ BASE_PATH }}/images/fsq.png" alt="4sq" height="500">
 
The Foursquare API allows you to search for venues within a specified radius around a latitude, longitude point and will return up to a maximum of 50 venues for a single call. I specified the venue category to be the category ID corresponding to "Food", and performed a few thousand calls to the API over a grid of latitude, longitude defined by the locations of Census tracts in Manhattan. Since there are only about 280 Census tracts in Manhattan, I interpolated extra latitude, longitude points between these values to create a finer grid. Finally because the API only allows 2500 calls per hour, I added a condition so the code would pause to prevent exceeding the API limit. I gathered data for around 10,500 restaurants.

To create the text corpus of restaurant comments I concatenated all the comment strings for each restaurant into a single long string: this became the "document" for a restaurant. Any restaurants with zero comments were discarded.

To draw out the most important features of each restaurant I calculated the term frequency-inverse document frequency (TF-IDF) statistic on the corpus of restaurant comments. I chose TF-IDF because the final statistical weight of each term includes the inverse document frequency. This is important so terms that appear many times in only a few documents are still weighted highly (e.g. a popular but generally uncommon food item that is a speciality in only a small subset of restaurants). I chose to include uni-grams (single word) and bi-grams (two words together) to include nouns that are made up of one or two words. I stopped at bi-grams because the complexity of matrix calculations goes as //(N^3//), where here //(N//) would be the number distinct keywords in this case. Because it is unlikely there are many tri-grams directly relevant to my use case (searching for mostly food/drink related nouns) including tri-grams would mostly increase noise in the keyword terms while drastically increasing the compute time.

The keyword terms in the resulting restaurant - term matrix needed to be cleaned: terms which were numbers or words in cyrillic were removed via regular expression search, and singular/plural pairs of the same word were aggregated.

I then clustered the restaurants based upon their term weight values. The idea of this is to refine the extremely noisy categorisation of the restaurants by Foursquare users, down from over 200 separate categories (ones that included seemingly non-restaurant things like "Museum" and "Pet Store") to a more manageable number with a more nuanced category not depending soley on cuisine type. I used k means to find 20 clusters based upon the distance between restaurants in TF-IDF term-weight space being defined by the cosine similarity. I chose k means to begin with because of its simplicity and also because I had a good idea of what the number of k clusters should be a priori. Manually assigning the 200+ Foursquare categories into restaurants with different cuisine types (or into "other" type) resulted in about 20 reasonably broad categories. I trialed a few different values of k between 5 and 25.

I validated the clusters (and the choice of k=20) by studying the frequencies of the top Foursquare categories, the top weighted terms, and the price distribution of all restaurants in the cluster. The clusters came out as shown in the following list. The head of the list gives the most frequent Foursquare category in the cluster, the five words after the colon are the top five terms in that cluster:

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

How the clusters were labelled:

After studying the distribution of the most frequent Foursquare restaurant categories, the most frequent keywords and the price distribution of restaurants in each cluster, usually the most frequent Foursquare category was assigned to be the "label" of that cluster. A few exceptions are described (and the final label is shown in brackets in the list above):

* the "Pizza" labels were assigned by looking at the price differences between the two clusters: one was clearly skewed low and the other high. 
* The "Starbucks" label was assigned because 99% of all the restaurants in this cluster were Starbucks, while zero of the restaurants in the other coffee category were Starbucks.
* The "Fancy" label was chosen because the most frequent restaurant types were fairly evenly spread over French, New American, Steakhouse and Seafood (restaurant types that tend to be at a higher price point), and the top keywords were more expensive items also
* The two clusters that were labelled "other" were discarded from further analysis because these refer to just generic snack food places, take-out and markets.

The plot below shows the first two principal components of the restaurant - term matrix for restaurants in the top 5 largest clusters only, and the colors show the Foursquare defined category. The PCA analysis demonstrates that the clustering does pull out the distinct categories: the pink/blue branch leading up to the top right corner are Asian restaurants (Chinese and Thai), the top right corner is a mixture of the "Fancy" restaurants (French, Steakhouse, New American, Seafood), the bottom center area is coffee shops, cafes and bakeries. The central area is more sparse and contains types such as Deli/Bodega and Sandwich places which tend to end up in the clusters that are discarded. The clustering of restaurants based upon the keywords is noisy, and expectedly so, because restaurants will always have many features that overlap with other types of restaurants.

![pca]({{ BASE_PATH }}/images/pca.png) 


Switching gears to the public data, I used the demographics data from the American Community Survey. I used the United State's Census Bureau's [API](https://www.census.gov/developers/), querying it for all the relevant fields (age, gender, median income, education) and specifying only New York County (county FIPS code = '061') and had the data returned at the level of a census tract. A census tract is a small subdivision of a county which contains around 1000 to 4000 residents.

Here's a taste of the census data. A neighborhood with a larger amount of millennials generally also has a larger median income. The fraction of women in neighborhoods is consistently higher than the fraction of men, as can be seen from the larger density of red points greater than 0.5 (matching the common understanding of NYC!)

![census1]({{ BASE_PATH }}/images/census_plot1.png) | ![census2]({{ BASE_PATH }}/images/men_women.png)

I also downloaded the GeoJSON file containing the details of the New York State census tracts. I created a new file by pulling out only the New York Country tracts and additionally removing unuseful tracts (those containing zero or anomolously low populations, e.g. tracts over public parks etc) to reduce the file to only the areas I wanted to focus on.

Finally, to relate the data of the restaurants (keyword-frequency and metadata) to the census tracts I used the Shapely package. Given a coordinate point and a geometry defined by a polygon in the GeoJSON file, the Shapely.geometry package can return whether or not the coordinate point lies within the polygon. There were some failures when I used this to assign a census tract to a restaurant location, but mostly they were due to the restaurant location lying exactly on a border between two census tracts. To resolve this failure I simply assigned the restaurant to the tract with the physically closest central coordinate.

# Example Analysis

Let's compare neighborhoods that have an excess of Baby Boomers or an excess of Millennials:

![boomer map]({{ BASE_PATH }}/images/boommills.png)

The Baby Boomers are mostly clustered around the east side of the park, and the Millennials tend to be more downtown/lower Manhattan. We can also compare the distribution of restaurant categories in both neighborhoods:

![millennial map]({{ BASE_PATH }}/images/restos.png)

The black outline bars show the baseline distribution of restaurant categories over all of Manhattan (and so are identical in left and right plots). The pink bars show the distribution of the restaurant categories in each type of neighborhood (left Baby Boom, right Millennial). Many things make intuitive sense, e.g. the excess of bars in the Millennial neighborhoods and the corresponding decrement in Baby Boomer neighborhoods. In particular we notice an excess of "fancy" restaurants in Baby Boomer neightborhoods coupled with a decrement of Mexican restaurants: *perhaps an untapped market in these neighborhoods could be an upscale Mexican restaurant?*

# The Tool

A user types an address located in Manhattan, and a series of diagnostics are returned including an emphasis on what is unusual or an outlier about a neightborhood with significance estimates and a list of other areas containing similar neighborhoods. For example typing `45 W 25th Street` returns:

    You are in Hudson Yards-Chelsea-Flatiron-Union Square

    Neighborhood is in top 88% for boomers
    Neighborhood is in top 3% for millennials
    Neighborhood is in top 77% for women
    Neighborhood is in top 22% for men
    Neighborhood is in top 3% for highly educated
    Neighborhood is in top 9% for income
    
![histogram of outliers]({{ BASE_PATH }}/images/demos.png)

    Total number of restaurants in neighborhood = 84
    
    Significant excess of “BAR” (p = 0.008)
    Significant excess of “SANDWICHES” (p = 0.000)
    Significant excess of “FANCY” (p = 0.000)
    
![reataurant distribution]({{ BASE_PATH }}/images/restos_eg.png)
    
    These areas contain neighborhoods that are similar:
    Hudson Yards-Chelsea-Flatiron-Union Square
    Midtown-Midtown South
    West Village
    
    Top restaurant keywords for this tract are:
    beer
    bowl
    sandwich
    starbucks
    pizza

# In summary

My project provides an overview of the customer demographics in an area. It enables the identification of the excess or decrement of any restaurant category in a neighborhood compared to the more global area. It also provides a list of areas containing neighborhoods with the most similar conditions. All this is in order to provide information for making data driven insights.


