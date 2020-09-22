#!/usr/bin/env python
# coding: utf-8

# ### `Project - MovieLens Data Analysis`
# 
# The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. The data is widely used for collaborative filtering and other filtering solutions. However, we will be using this data to act as a means to demonstrate our skill in using Python to “play” with data.
# 
# 
# ### `Objective:`
# - To implement the techniques learnt as a part of the course.
# 
# ### `Learning Outcomes:`
# - Exploratory Data Analysis
# 
# - Visualization using Python
# 
# - Pandas – groupby, merging 
# 
# 
# ### `Domain` 
# - Internet and Entertainment
# 
# **Note that the project will need you to apply the concepts of groupby and merging extensively.**

# ### `Datasets Information:`
# 
# 
# *rating.csv:* It contains information on ratings given by the users to a particular movie.
# - user id: id assigned to every user
# - movie id: id assigned to every movie
# - rating: rating given by the user
# - timestamp: Time recorded when the user gave a rating
# 
# 
# 
# *movie.csv:* File contains information related to the movies and their genre.
# - movie id: id assigned to every movie
# - movie title: Title of the movie
# - release date: Date of release of the movie
# - Action: Genre containing binary values (1 - for action 0 - not action)
# - Adventure: Genre containing binary values (1 - for adventure 0 - not adventure)
# - Animation: Genre containing binary values (1 - for animation 0 - not animation)
# - Children’s: Genre containing binary values (1 - for children's 0 - not children's)
# - Comedy: Genre containing binary values (1 - for comedy 0 - not comedy)
# - Crime: Genre containing binary values (1 - for crime 0 - not crime)
# - Documentary: Genre containing binary values (1 - for documentary 0 - not documentary)
# - Drama: Genre containing binary values (1 - for drama 0 - not drama)
# - Fantasy: Genre containing binary values (1 - for fantasy 0 - not fantasy)
# - Film-Noir: Genre containing binary values (1 - for film-noir 0 - not film-noir)
# - Horror: Genre containing binary values (1 - for horror 0 - not horror)
# - Musical: Genre containing binary values (1 - for musical 0 - not musical)
# - Mystery: Genre containing binary values (1 - for mystery 0 - not mystery)
# - Romance: Genre containing binary values (1 - for romance 0 - not romance)
# - Sci-Fi: Genre containing binary values (1 - for sci-fi 0 - not sci-fi)
# - Thriller: Genre containing binary values (1 - for thriller 0 - not thriller)
# - War: Genre containing binary values (1 - for war 0 - not war)
# - Western: Genre containing binary values (1 - for western - not western)
# 
# 
# 
# *user.csv:* It contains information of the users who have rated the movies.
# - user id: id assigned to every user
# - age: Age of the user
# - gender: Gender of the user
# - occupation: Occupation of the user
# - zip code: Zip code of the use
# 
# 
# **`Please provide you insights wherever necessary.`**

# ### 1. Import the necessary packages - 2.5 marks

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# ### 2. Read the 3 datasets into dataframes - 2.5 marks

# In[2]:


ratings = pd.read_csv('Data.csv')
movie = pd.read_csv('item.csv')
user = pd.read_csv('user.csv')


# In[3]:


ratings.head()


# In[4]:


movie.head()


# In[5]:


user.head()
print(user['user id'].count())
user2 = user.groupby(['gender']).nunique()
user2.drop(['age', 'occupation', 'zip code'], axis=1, inplace=True)
user2


# In[ ]:





# ### 3. Apply info, shape, describe, and find the number of missing values in the data - 5 marks
#  - Note that you will need to do it for all the three datasets seperately

# In[6]:


movie.info()


# In[7]:


movie.shape


# In[8]:


movie.describe()


# In[9]:


movie.info()


# In[10]:


ratings.info()


# In[11]:


ratings.shape


# In[12]:


ratings.describe()


# In[13]:


user.info()


# In[14]:


user.shape


# In[15]:


user.describe()


# #### Based on the above info, all three datasets have no missing values.

# ### 4. Find the number of movies per genre using the item data - 2.5 marks

# In[16]:


genre = movie.copy()
genre.head()

genre2 = genre.drop(['movie title', 'movie id', 'unknown', 'release date'], axis=1).sum()
genre2


# ### 5. Drop the movie where the genre is unknown - 2.5 marks

# In[17]:


movie['unknown'].any()


# In[18]:


movie[movie['unknown'] == 1]


# In[19]:


movie.drop(movie[movie.unknown == 1].index, inplace=True)


# In[ ]:





# ### 6. Find the movies that have more than one genre - 5 marks
# 
# hint: use sum on the axis = 1
# 
# Display movie name, number of genres for the movie in dataframe
# 
# and also print(total number of movies which have more than one genres)

# In[20]:


# Dropping rows to create new dataframe

genre2 = genre.drop(['movie id', 'release date', 'unknown'], axis=1)

# Creating new column with number of genres for each movie

genre2['Genres'] = genre2.sum(axis=1)

# Setting indexes as movie titles

genre2.set_index('movie title', inplace=True)

# Dropping all columns except new "Genres" column

genre2.drop(genre2.iloc[:, 0:-1], inplace=True, axis=1)

# Dropping all rows with titles having less than 2 genres

genre2.drop(genre2[genre2.Genres < 2].index, inplace=True)

genre2.sort_values(by='Genres', ascending=False)

#genre2['Genres'].count()


# In[21]:


rows = len(genre2['Genres'])

print("Total number of movies which have more than one genre: ", rows)


# ### 7. Univariate plots of columns: 'rating', 'Age', 'release year', 'Gender' and 'Occupation' - 10 marks
# 
# *HINT: Use distplot for age. Use lineplot or countplot for release year.*
# 
# *HINT: Plot percentages in y-axis and categories in x-axis for ratings, gender and occupation*
# 
# *HINT: Please refer to the below snippet to understand how to get to release year from release date. You can use str.split() as depicted below or you could convert it to pandas datetime format and extract year (.dt.year)*

# In[22]:


a = 'My*cat*is*brown'
print(a.split('*')[3])

#similarly, the release year needs to be taken out from release date

#also you can simply slice existing string to get the desired data, if we want to take out the colour of the cat

print(a[10:])
print(a[-5:])


# In[23]:


movie2 = movie.copy()

movie2['release date'] = movie2['release date'].apply(lambda x: x.split('-')[2])


# In[24]:


movie2


# In[25]:


ratings.head()


# In[26]:


user.head()


# In[27]:


import matplotlib.ticker as mtick

perc = np.linspace(0,100,len(ratings['rating']))

ax = sb.distplot(ratings['rating'])

ax.yaxis.set_major_formatter(mtick.PercentFormatter())


# In[28]:


sb.distplot(user['age']);


# In[29]:


plt.figure(figsize=(15,5));
ax = sb.countplot(movie2['release date']);
plt.xticks(rotation=70);


# In[30]:


ax = sb.countplot(x = user['gender'])


# In[31]:


ax = sb.countplot(user['occupation']);
plt.xticks(rotation='vertical');


# ### 8. Visualize how popularity of genres has changed over the years - 10 marks
# 
# Note that you need to use the **percent of number of releases in a year** as a parameter of popularity of a genre
# 
# Hint 1: You need to reach to a data frame where the release year is the index and the genre is the column names (one cell shows the number of release in a year in one genre) or vice versa. (Drop unnecessary column if there are any)
# 
# Hint 2: Find the total number of movies release in a year(use `sum(axis=1)` store that value in a new column as 'total'). Now divide the value of each genre in that year by total to get percentage number of release in a particular year.
# `(df.div(df['total'], axis= 0) * 100)`
# 
# Once that is achieved, you can either use univariate plots or can use the heatmap to visualise all the changes over the years 
# in one go. 
# 
# Hint 3: Use groupby on the relevant column and use sum() on the same to find out the number of releases in a year/genre.  

# In[32]:


pop = movie2.copy()


# In[33]:


pop.drop(['movie id', 'movie title', 'unknown'], axis=1, inplace=True)


# In[34]:


pop = pop.groupby(['release date']).sum()
pop


# In[35]:


pop['total'] = pop.sum(axis=1)


# In[36]:


pop = pop.div(pop['total'], axis=0) * 100

columns = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

pop2 = pop[columns]
pop2


# In[37]:


fig, ax = plt.subplots(figsize=(20,15))

sb.heatmap(pop2, cmap='viridis', annot=True)


# ### 9. Find the top 25 movies according to average ratings such that each movie has number of ratings more than 100 - 10 marks
# 
# Hints : 
# 
# 1. Find the count of ratings and average ratings for every movie.
# 2. Slice the movies which have ratings more than 100.
# 3. Sort values according to average rating such that movie which highest rating is on top.
# 4. Select top 25 movies.
# 5. You will have to use the .merge() function to get the movie titles.
# 
# Note: This question will need you to research about groupby and apply your findings. You can find more on groupby on https://realpython.com/pandas-groupby/.

# In[38]:


# Merging the movie and ratings dataframes

top = pd.merge(movie, ratings, how='inner', on='movie id')

# Dropping extra columns

top.drop(['release date', 'unknown'], axis=1, inplace=True)
top.drop(top.iloc[:, 2:-3], axis=1, inplace=True)
top


# In[39]:


# Calling groupby on 'movie title' column and counting # of ratings per movie

rating_count = top.groupby(['movie title']).count()

# Dropping extra columns

rating_count.drop(['user id', 'timestamp'], axis=1, inplace=True)
rating_count.drop(['movie id'], axis=1, inplace=True)

rating_count


# In[40]:


# Averaging ratings for each movie

average = top.groupby(['movie title']).mean()
average


# In[41]:


# Slicing movies with more than 100 ratings

hundred = rating_count[rating_count['rating'] > 100]
hundred


# In[42]:


hundred_mean = hundred.merge(average, how='left', on='movie title')


# In[43]:


hundred_mean


# In[44]:


# Sorting movies so that highest rating is on top

hundred_mean = hundred_mean.sort_values(by='rating_y', ascending=False)


# In[45]:


hundred_mean


# In[46]:


# Selecting the top 25 movies

print("The top 25 movies are:")
hundred_mean.head(25)


# ### 10. See gender distribution across different genres check for the validity of the below statements - 10 marks
# 
# * Men watch more drama than women
# * Women watch more Sci-Fi than men
# * Men watch more Romance than women
# 
# **compare the percentages**

# 1. Merge all the datasets
# 
# 2. There is no need to conduct statistical tests around this. Just **compare the percentages** and comment on the validity of the above statements.
# 
# 3. you might want ot use the .sum(), .div() function here.
# 
# 4. Use number of ratings to validate the numbers. For example, if out of 4000 ratings received by women, 3000 are for drama, we will assume that 75% of the women watch drama.

# In[47]:


gender = pd.merge(user, ratings, how='inner', on='user id')
gender


# In[48]:


gender = pd.merge(gender, movie, how='inner', on='movie id')
gender


# In[89]:


gender_dummies = pd.get_dummies(gender, prefix='Gender', columns=['gender'])

gender2 = gender_dummies[['rating', 'Gender_M', 'Gender_F', 'Drama', 'Romance', 'Sci-Fi']]

gender_grouped = gender2.groupby(['Gender_M', 'Gender_F']).sum()

#gender_grouped.sum()

gender_grouped


# In[90]:


males_total = gender_dummies['Gender_M'].sum()
females_total = gender_dummies['Gender_F'].sum()

m_drama_pct = int(gender_grouped['Drama'][1]) / males_total
f_drama_pct = int(gender_grouped['Drama'][0]) / females_total

m_romance_pct = int(gender_grouped['Romance'][1]) / males_total
f_romance_pct = int(gender_grouped['Romance'][0]) / females_total

m_scifi_pct = int(gender_grouped['Sci-Fi'][1]) / males_total
f_scifi_pct = int(gender_grouped['Sci-Fi'][0]) / females_total

print(m_scifi_pct, f_scifi_pct)


# In[91]:


if f_drama_pct < m_drama_pct:
    print("Men watch more drama than women.")
else:
    print("Women watch more drama than men.")
    
if f_romance_pct < m_romance_pct:
    print("Men watch more romance than women.")
else:
    print("Women watch more romance than men.")

if f_scifi_pct < m_scifi_pct:
    print("Men watch more sci-fi than women.")
else:
    print("Women watch more sci-fi than men.")


# In[ ]:




