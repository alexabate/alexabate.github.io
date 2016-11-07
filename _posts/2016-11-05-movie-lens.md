---
layout: post
title: "Movie Lens!"
tags:
    - python
    - notebook
---

 
## The MovieLens data set

[This is a classic data set](http://grouplens.org/datasets/movielens/100k/) that is used by people who want to try out
recommender system approaches. It was set up in 1997 by a group at the
University of Minnesota who wanted to gather research data on personalised
recommendations.

We have the following three tables:

* the ratings users gave to movies
* data on the users (age, gender, occupation, zip code)
* data on the movies (title, release data, genre(s))

In this version of the data set we have 100,000 ratings of nearly 1700 movies
from nearly 1000 users. 
 
# User ratings for movies

This table is the 'meat' required for any recommender approach: the actual
ratings 


{% highlight python %}
names = ['user_id', 'movie_id', 'rating', 'timestamp']
user_item_ratings = pd.read_csv('u.data', sep='\t', names=names)
user_item_ratings.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>


{% highlight python %}
number_of_unique_users = len(user_item_ratings['user_id'].unique())
number_of_unique_movies = len(user_item_ratings['movie_id'].unique())
number_of_ratings = len(user_item_ratings)
print "Number of unique users =", number_of_unique_users
print "Number of unique movies =", number_of_unique_movies
print "Number of ratings =", number_of_ratings
{% endhighlight %}

    Number of unique users = 943
    Number of unique movies = 1682
    Number of ratings = 100000



{% highlight python %}
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.hist(user_item_ratings['rating'], bins=[0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1, 4.9, 5.1])
ax.set_xlabel('movie rating', fontsize=24)

mean_rating = user_item_ratings['rating'].mean()
print "Mean rating =", mean_rating
{% endhighlight %}

    Mean rating = 3.52986


 
![png]({{ BASE_PATH }}/images/movie-lens!_5_1.png) 



{% highlight python %}
# Number of ratings per movie
num_ratings_per_movie = user_item_ratings['movie_id'].value_counts().values

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.hist(num_ratings_per_movie, 100)
ax.set_xlabel('Number of ratings per movie', fontsize=24)

print len(np.where(num_ratings_per_movie<2)[0]), "movies were rated by only 1 user"
{% endhighlight %}

    141 movies were rated by only 1 user


 
![png]({{ BASE_PATH }}/images/movie-lens!_6_1.png) 



{% highlight python %}
# mean ratings from each user
mean_rating_of_user = user_item_ratings.groupby('user_id').apply(lambda x: 
                                                                 x['rating'].mean())

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.hist(mean_rating_of_user)
ax.set_xlabel('mean rating given by each user', fontsize=24)
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/movie-lens!_7_1.png) 

 
# Movie data 

{% highlight python %}
names = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
         'IMDb_URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
         'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
         'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
         'Thriller' , 'War' , 'Western']
movies = pd.read_csv('u.item', sep='|', names=names).set_index('movie_id')
movies.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_title</th>
      <th>release_date</th>
      <th>video_release_date</th>
      <th>IMDb_URL</th>
      <th>unknown</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>...</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Toy Story (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Toy%20Story%2...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GoldenEye (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?GoldenEye%20(...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Four Rooms (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Four%20Rooms%...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Get Shorty (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Get%20Shorty%...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Copycat (1995)</td>
      <td>01-Jan-1995</td>
      <td>NaN</td>
      <td>http://us.imdb.com/M/title-exact?Copycat%20(1995)</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>


{% highlight python %}
genres = ['unknown' , 'Action' , 'Adventure' , 'Animation' ,
         'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
         'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
         'Thriller' , 'War' , 'Western']

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
movies[genres].sum().sort_values().plot(kind='barh', stacked=False, ax=ax)

{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/movie-lens!_10_1.png) 

 
# Users 


{% highlight python %}
names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=names).set_index('user_id')
users.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>occupation</th>
      <th>zip_code</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>85711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>F</td>
      <td>other</td>
      <td>94043</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>M</td>
      <td>writer</td>
      <td>32067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>M</td>
      <td>technician</td>
      <td>43537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>F</td>
      <td>other</td>
      <td>15213</td>
    </tr>
  </tbody>
</table>
</div>


{% highlight python %}
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
users['gender'].value_counts().plot(kind='barh', stacked=False, ax=ax)
ax.set_yticklabels(['Male', 'Female'], fontsize=20)
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/movie-lens!_13_1.png) 


{% highlight python %}
grouped_by_gender = users.groupby(["occupation","gender"]).size().unstack(
                           "gender").fillna(0)

frac_gender = grouped_by_gender.divide(grouped_by_gender.sum(axis=1), axis='rows')
 

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
frac_gender.sort_values('M').plot(kind='barh', stacked=False, ax=ax, 
                       color=sns.color_palette('hls'))
ax.legend(fontsize=24, loc='lower right')
ax.set_ylabel('')
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/movie-lens!_14_1.png) 


{% highlight python %}
def outline_boxes_with_color(bp, col, color):
    """Change all lines on boxplot, the box outline, the median line, the whiskers and
       the caps on the ends to be 'color'
       
       @param bp       handle of boxplot
       @param col      column
       @param color    color to change lines to
    """ 
    
    plt.setp(bp[col]['boxes'], color=color)  
    plt.setp(bp[col]['whiskers'], color=color)  
    plt.setp(bp[col]['medians'], color=color)  
    plt.setp(bp[col]['caps'], color=color) 
    
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
col = 'age'
bp = users.boxplot(column=[col], return_type='dict', 
                         by='occupation', ax=ax)
outline_boxes_with_color(bp, col, 'black')
ax.set_xlabel('', fontsize=24)
ax.set_ylabel('Age', fontsize=24)
ax.set_title('')
plt.xticks(rotation='vertical', fontsize=18)
fig.suptitle('')
{% endhighlight %}


 
![png]({{ BASE_PATH }}/images/movie-lens!_15_1.png) 

 
## User-item matrix


{% highlight python %}
ratings_matrix = np.zeros((number_of_unique_users, number_of_unique_movies))

for index, row in user_item_ratings.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    rating = row['rating']
    
    ratings_matrix[user_id-1, movie_id-1] = rating

{% endhighlight %}


{% highlight python %}
sparsity = 100.*float(len(ratings_matrix.nonzero()[0]))/ratings_matrix.size
print 'Sparsity of the user-item matrix ={0:4.1f}%'.format(sparsity)

{% endhighlight %}

    Sparsity of the user-item matrix = 6.3%

 
## Step 0: Most basic recommendation

Predict rating to always be the mean movie rating! This is the baseline we seek
to improve upon. 


{% highlight python %}
rmse_benchmark = np.sqrt(pow(user_item_ratings['rating']-mean_rating, 2).mean())
print "Maximum root-mean-square error = {:4.2f}".format(rmse_benchmark)
{% endhighlight %}

    Maximum root-mean-square error = 1.13

 
## Step 1: Rating prediction based upon user-user similarity

We define "similarity" using the cosine similarity metric. Imagining each user's
set of ratings as a vector, the cosine similarity of two users is simply the
cosine of the angle between their two vectors. This is given by the dot product
of the two vectors divided by their magnitudes:

$$ \mbox{similarity}(u_i, u_j) = \frac{\textbf{r}_{u_i} \cdot
\textbf{r}_{u_j}}{\mid \textbf{r}_{u_i} \mid \mid \textbf{r}_{u_j} \mid} $$

And for ease i'll simply do a 'leave one out' approach: for every user I will
use the other users' ratings to predict that user's ratings, and then calculate
an overall root-mean-square-error on those predictions.

The code below is also slow because it's not taking full advantage of matrix
operations, but **THIS CODE IS FOR ILLUSTRATION ONLY!**
 


{% highlight python %}
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the similarity score between users
user_user_similarity = cosine_similarity(ratings_matrix)
user_user_similarity.shape
{% endhighlight %}




    (943, 943)


 
# Predict a rating as exactly the rating given by the most similar user 



{% highlight python %}

sqdiffs = 0
num_preds = 0
cnt_no_other_ratings = 0

# for each user
for user_i, u in enumerate(ratings_matrix):

    # movies user HAS rated
    i_rated = np.where(u>0)[0]
    
    # for each rated movie: find most similar user who HAS ALSO rated this movie
    for imovie in i_rated:
                
        # all users that have rated imovie (includes user of interest)
        i_has_rated = np.where(ratings_matrix[:, imovie]>0)[0]
        
        # remove the current user 
        iremove = np.argmin(abs(i_has_rated - user_i)) 
        i_others_have_rated = np.delete(i_has_rated, iremove)
        
        
        # find most similar user that has also rated imovie to current user
        try:
            i_most_sim = np.argmax(user_user_similarity[user_i, i_others_have_rated])
        except:
            cnt_no_other_ratings += 1
            continue
        
        # prediction error
        predicted_rating = ratings_matrix[i_others_have_rated[i_most_sim], imovie]
        actual_rating = ratings_matrix[user_i, imovie]
        
        sqdiffs += pow(predicted_rating-actual_rating, 2.)
        num_preds += 1


rmse_cossim = np.sqrt(sqdiffs/num_preds)

print "Failed to make", cnt_no_other_ratings ,"predictions"
print "Number of predictions made =", num_preds
print "Root mean square error =", rmse_cossim
{% endhighlight %}

    Failed to make 141 predictions
    Number of predictions made = 99859
    Root mean square error = 1.30217928718


Erm, it got worse! Let's try something more sensible. (The failed predictions occur because that movie was only rated by a single user.)


# Predict a rating as the weighted mean of all ratings

Weighted by the user similarities ... 

$$ \hat{r}_{u, i} = \frac{ \sum_{u_j} \mbox{similarity}(u_u,
u_j)r_{u_j}}{\sum_{u_j} \mbox{similarity}(u_u, u_j)}$$


{% highlight python %}
sqdiffs = 0
num_preds = 0

# to protect against divide by zero issues
eps = 1e-6

cnt_no_sims = 0

# for each user
for user_i, u in enumerate(ratings_matrix):

    # movies user HAS rated
    i_rated = np.where(u>0)[0]
    
    # for each rated movie: find users who HAVE ALSO rated this movie
    for imovie in i_rated:
                
        # all users that have rated imovie (includes user of interest)
        i_has_rated = np.where(ratings_matrix[:, imovie]>0)[0]
        
        # remove the current user 
        iremove = np.argmin(abs(i_has_rated - user_i)) 
        i_others_have_rated = np.delete(i_has_rated, iremove)
        
        
        # rating is weighted sum of all ratings, weights are cosine sims
        ratings = ratings_matrix[i_others_have_rated, imovie]
        sims = user_user_similarity[user_i, i_others_have_rated]
        
        norm = np.sum(sims)
        if norm==0:
            cnt_no_sims += 1
            norm = eps
                
        predicted_rating = np.sum(ratings*sims)/norm
        
        
        # prediction error
        actual_rating = ratings_matrix[user_i, imovie]
        
        sqdiffs += pow(predicted_rating-actual_rating, 2.)
        num_preds += 1
        

rmse_cossim = np.sqrt(sqdiffs/num_preds)

print cnt_no_sims, "movies had only one user rating" 
print "Number of predictions made =", num_preds
print "Root mean square error =", rmse_cossim
{% endhighlight %}

    141 movies had only one user rating
    Number of predictions made = 100000
    Root mean square error = 1.01584998447

 
It improved! Can we do any better by only counting the top $$n$$ most similar
users in the weighted sum? 


{% highlight python %}

def rmse_topN(topN):
    """Return the root-mean-square-error given value topN
       for using the 'top N' most similar users in predicting
       the rating
    """

    sqdiffs = 0
    num_preds = 0

    # to protect against divide by zero issues
    eps = 1e-6

    cnt_no_sims = 0

    # for each user
    for user_i, u in enumerate(ratings_matrix):

        # movies user HAS rated
        i_rated = np.where(u>0)[0]

        # for each rated movie: 
        for imovie in i_rated:

            # all users that have rated imovie (includes user of interest)
            i_has_rated = np.where(ratings_matrix[:, imovie]>0)[0]

            # remove the current user 
            iremove = np.argmin(abs(i_has_rated - user_i)) 
            i_others_have_rated = np.delete(i_has_rated, iremove)


            # rating is weighted sum of all ratings, weights are cosine sims
            ratings = ratings_matrix[i_others_have_rated, imovie]
            sims = user_user_similarity[user_i, i_others_have_rated]

            # only want top n sims
            most_sim_users = sims[np.argsort(sims*-1)][:topN]
            most_sim_ratings = ratings[np.argsort(sims*-1)][:topN]

    #         if user_i == 0:
    #             break

            norm = np.sum(most_sim_users)
            if norm==0:
                cnt_no_sims += 1
                norm = eps

            predicted_rating = np.sum(most_sim_ratings*most_sim_users)/norm


            # prediction error
            actual_rating = ratings_matrix[user_i, imovie]

            sqdiffs += pow(predicted_rating-actual_rating, 2.)
            num_preds += 1

    #     if user_i == 0:
    #         break

    rmse_cossim = np.sqrt(sqdiffs/num_preds)

    print "Using top", topN , "most similar users to predict rating"
    print "Number of predictions made =", num_preds
    print "Root mean square error =", rmse_cossim , '\n'
    return rmse_cossim

topN_trials = [2, 10, 25, 50, 75, 100, 300]
rmse_results = []
for topN in topN_trials:
    rmse_results.append(rmse_topN(topN))
{% endhighlight %}

    Using top 2 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.14548307354 
    
    Using top 10 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.01567130006 
    
    Using top 25 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.0023325746 
    
    Using top 50 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.0039703331 
    
    Using top 75 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.00673516576 
    
    Using top 100 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.00894503162 
    
    Using top 300 most similar users to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.01523721106 
    



{% highlight python %}
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(topN_trials, rmse_results, label='using top N')
xlims = ax.get_xlim()
ax.plot(xlims, [rmse_benchmark, rmse_benchmark], color='black', 
        linestyle='dotted', label='benchmark')
ax.plot(xlims, [rmse_cossim, rmse_cossim], color='blue', linestyle='dotted',
        label='using all')

ax.set_xlabel('use top N most similar users', fontsize=24)
ax.set_ylabel('RMSE', fontsize=24)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=20)
{% endhighlight %}






 
![png]({{ BASE_PATH }}/images/movie-lens!_29_1.png) 

 
Yes it improves if we use $$10 < N <300 $$ top users, with $$N=25$$ looking like it gives
the best improvement.

## Step 2: item-item similarity

We can do exactly the same process for items instead of users; this time treating an item as
a vector of ratings and calculate a similarity between two items in the same
manner using cosine similarity. 


{% highlight python %}
# Calculate the similarity score between ITEMS: note the transpose
item_item_similarity = cosine_similarity(ratings_matrix.T)
item_item_similarity.shape
{% endhighlight %}




    (1682, 1682)



{% highlight python %}
def rmse_topN_items(topN):
    """Return the root-mean-square-error given value topN
       for using the 'top N' most similar ITEMS in predicting
       the rating
    """

    sqdiffs = 0
    num_preds = 0

    # to protect against divide by zero issues
    eps = 1e-6

    cnt_no_sims = 0

    # for each item
    for item_i, u in enumerate(ratings_matrix.T):

        # users item HAS ratings by
        i_rated = np.where(u>0)[0]

        # for each user that rated: 
        for iuser in i_rated:

            # all movies that have been rated by iuser
            i_has_rated = np.where(ratings_matrix[iuser, :]>0)[0]

            # remove the current movie 
            iremove = np.argmin(abs(i_has_rated - item_i)) 
            i_others_have_rated = np.delete(i_has_rated, iremove)


            # rating is weighted sum of all ratings, weights are cosine sims
            ratings = ratings_matrix[iuser, i_others_have_rated]
            sims = item_item_similarity[i_others_have_rated, item_i]

            # only want top n sims
            most_sim_users = sims[np.argsort(sims*-1)][:topN]
            most_sim_ratings = ratings[np.argsort(sims*-1)][:topN]


            norm = np.sum(most_sim_users)
            if norm==0:
                cnt_no_sims += 1
                norm = eps

            predicted_rating = np.sum(most_sim_ratings*most_sim_users)/norm


            # prediction error
            actual_rating = ratings_matrix[iuser, item_i]

            sqdiffs += pow(predicted_rating-actual_rating, 2.)
            num_preds += 1


    rmse_cossim = np.sqrt(sqdiffs/num_preds)

    print "Using top", topN , "most similar movies to predict rating"
    print "Number of predictions made =", num_preds
    print "Root mean square error =", rmse_cossim , '\n'
    return rmse_cossim

topN_trials = [2, 10, 25, 50, 75, 100, 300]
rmse_item_results = []
for topN in topN_trials:
    rmse_item_results.append(rmse_topN_items(topN))
{% endhighlight %}

    Using top 2 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.0627135524 
    
    Using top 10 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 0.959118494932 
    
    Using top 25 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 0.965039905069 
    
    Using top 50 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 0.980703964443 
    
    Using top 75 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 0.989516609768 
    
    Using top 100 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 0.995218394229 
    
    Using top 300 most similar movies to predict rating
    Number of predictions made = 100000
    Root mean square error = 1.01019290087 
    

{% highlight python %}
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(topN_trials, rmse_item_results, label='item-item')
ax.plot(topN_trials, rmse_results, label='user-user')

xlims = ax.get_xlim()
ax.plot(xlims, [rmse_benchmark, rmse_benchmark], color='black', 
        linestyle='dotted', label='benchmark')

ax.set_xlabel('use top N most similar', fontsize=24)
ax.set_ylabel('RMSE', fontsize=24)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=20)
{% endhighlight %}


 
![png]({{ BASE_PATH }}/images/movie-lens!_33_1.png) 

## Step 3: User bias!

Not all users rate movies in the same way, so it would be more useful if the
collaborative filtering looked at the relative difference between movie ratings
rather than the absolute values. E.g. look at the large scatter in the
distribution of the mean rating given by each user; some of this is coming from
noise (some users will have only rated 10 movies), but some is also coming from
the fact that some users will tend to consistently rate things higher than other
users. To account fo this we simply re-define the predicted rating
$$\hat{r_{u,i}}$$ for user $$u$$ and item $$i$$ as:


$$ \hat{r}_{u, i} = \bar{r}_{u} + \frac{ \sum_{u_j} \mbox{similarity}(u_u,
u_j)(r_{u_j}-\bar{r}_{u})}{\sum_{u_j} \mbox{similarity}(u_u, u_j)}$$
 

{% highlight python %}
topN = 25 # best value from before

sqdiffs = 0
num_preds = 0

# to protect against divide by zero issues
eps = 1e-6

cnt_no_sims = 0

# for each user
for user_i, u in enumerate(ratings_matrix):

    # movies user HAS rated
    i_rated = np.where(u>0)[0]

    # for each rated movie: 
    for imovie in i_rated:

        # all users that have rated imovie (includes user of interest)
        i_has_rated = np.where(ratings_matrix[:, imovie]>0)[0]

        # remove the current user 
        iremove = np.argmin(abs(i_has_rated - user_i)) 
        i_others_have_rated = np.delete(i_has_rated, iremove)


        # rating is weighted sum of all ratings, weights are cosine sims
        ratings = ratings_matrix[i_others_have_rated, imovie]
        sims = user_user_similarity[user_i, i_others_have_rated]

        # only want top n sims
        most_sim_users = sims[np.argsort(sims*-1)][:topN]
        most_sim_ratings = ratings[np.argsort(sims*-1)][:topN]

#         if user_i == 0:
#             break

        norm = np.sum(most_sim_users)
        if norm==0:
            cnt_no_sims += 1
            norm = eps

        predicted_rating = mean_rating + np.sum((most_sim_ratings-mean_rating)*most_sim_users)/norm


        # prediction error
        actual_rating = ratings_matrix[user_i, imovie]

        sqdiffs += pow(predicted_rating-actual_rating, 2.)
        num_preds += 1

#     if user_i == 0:
#         break

rmse_bias = np.sqrt(sqdiffs/num_preds)

print "Using top", topN , "most similar users to predict rating"
print "Number of predictions made =", num_preds
print "Root mean square error =", rmse_bias , '\n'
{% endhighlight %}

## Summary
 
Alright! So after trying the following, predict user $$i$$'s rating of movie $$j$$ as being:

* the **same** rating as the most similar user to user $$i$$ who has rated movie $$j$$ (result=bad)
* the weighted sum of ratings by **all** other users who have rated movie $$j$$. The weights are given by the other users' similarities to user $$i$$ (result=ok)
* the weighted sum of ratings by **the top k** most similar users to user $$i$$ who have also rated movie $$j$$ (result=ok)
* the weighted sum of ratings for **the top k** most similar movies to movie $$j$$ (result=best)

Using the top-10 most similar items with an item-item collaborative filtering approach seems to perform the best!

To be continued .... to play with one or more of: matrix factorisation, additional features! 

