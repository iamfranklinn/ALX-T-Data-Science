#!/usr/bin/env python
# coding: utf-8

# # Project - Wrangling and Analyze Data: WeRateDogs Twitter Dataset 

# ### INTRODUCTION
# 
# - Project Details
# 
# - Data wrangling, which consists of:
# - Gathering data
# From 'twitter-archive-enhanced.csv' file.
# From a link.
# From twitter API.
# 
# - Assessing data
# - Cleaning data
# - Storing, analyzing, and visualizing the wrangled data
# 
# - Reporting on:
# Data wrangling efforts.
# Data analyses and visualizations
# 
# 

# In[6]:


#Setting Up all libraries 
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import os
import json
from PIL import Image
from io import BytesIO
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# ## Data Gathering
# In the cell below, gather **all** three pieces of data for this project and load them in the notebook. **Note:** the methods required to gather each data are different.
# 1. Directly download the WeRateDogs Twitter archive data (twitter_archive_enhanced.csv)

# In[7]:


#twitter archive
twitter_arch = pd.read_csv('twitter-archive-enhanced.csv')


# 2. Use the Requests library to download the tweet image prediction (image_predictions.tsv)

# In[8]:


#Downloading the image predictions file for the image
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
with open(os.path.join('data/' + url.split('/')[-1]), 'wb') as file:
    file.write(response.content)


# In[9]:


# Loading the 'image-predictions.tsv' file
img_pred = pd.read_csv('data/image-predictions.tsv', sep='\t')


# 3. Use the Tweepy library to query additional data via the Twitter API (tweet_json.txt)

# In[11]:


#loading the tweets data into pandas df
with open('tweet-json.txt') as file:
    twitter_json = pd.read_json(file, lines= True, encoding = 'utf-8')


# ### Visual Assessment

# In[12]:


# Assessing the first 5 rows of the 'twitter-archive-enhanced.csv' file
twitter_arch.head()


# In[13]:


# Sampling the 'twitter-archive-enhanced.csv' file
twitter_arch.sample(10)


# In[14]:


# Getting the summarized details of each column of the 'twitter-archive-enhanced.csv' file
twitter_arch.info()


# Some columns are having missing values as well as wrong datatypes

# In[15]:


# Getting a summary statistics of the 'rating_numerator'and
# 'rating_denominator' columns of the'twitter-archive-enhanced.csv' file
twitter_arch[['rating_numerator', 'rating_denominator']].describe()


# The rating denominator should not be less or more than 10.

# In[16]:


# looking for numerators with wrong values in the 'twitter-archive-enhanced.csv' file
twitter_arch[twitter_arch.rating_numerator <= 5].loc[0:, :].sample(5)


# Some rows do have rating denominators to be less than 10

# In[17]:


# Checking for the unique names if the 'twitter-archive-enhanced.csv' file
twitter_arch.name.unique()


# Names such as 'a', 'all', 'his' etc cannot be dog names as seen in the name column

# In[18]:


# Checking the value counts of each unique name in the 'twitter-archive-enhanced.csv' file
twitter_arch.name.value_counts()


# It is observed that some of the names are in lower cases. Let's further find those names out.

# In[19]:


# Checking names that start with lower cases in the 'twitter-archive-enhanced.csv' file
twitter_arch[twitter_arch.name.str.islower()]


# # ASSESSMENT OF image-predictions.tsv FILE

# ## VISUAL ASSESSMENT

# In[20]:


# Checking the first five rows of the `image-prediction.tsv` file
img_pred.head()


# In[21]:


# Sampling five rows of the `image-prediction.tsv` file
img_pred.sample(5)


# What at all is img_num and what role is it playing in the dataset? p1, p2 and p3 columns seem to be about the same thing

# ## PROGRAMMATIC ASSESSMENT

# In[22]:


# Getting the summarized details of each column of the 'image-predictions.tsv' file
img_pred.info()


# There seem to be no null entry in this dataset.

# In[23]:


# Checking the value counts of each unique `img_num` in the 'image-predictions.tsv' file
img_pred.img_num.value_counts()


# The img_num column seem to have only values of 1 to 4.

# In[24]:


# Checking for the unique names in the `p1` column of the 'image-predictions.tsv' file
img_pred.p1.unique()


# Names in p1 columns as well as the p2 and p3 start with lower case letter.

# # ASSESSMENT OF tweet-json.txt FILE

# ## VISUAL ASSESSMENT

# In[25]:


# Checking the first five rows of the `tweet-json.txt` file
twitter_json.head()


# In[26]:


# Sampling five rows of the `tweet-json.txt` file
twitter_json.sample(5)


# The id column here seem to have a different name from the previous datasets.

# ## PROGRAMMATIC ASSESSMENT

# In[27]:


# Checking the info on the columns of the `tweet-json.txt` file
twitter_json.info()


# Are all these 30 columns relevant? Most of them also have missing values.

# In[28]:


# Checking the value counts of the languages used in the `tweet-json.txt` file
twitter_json.lang.value_counts()


# English language appears to be the most used language in the dataset.

# # OBSERVATIONS
# 
# The following observations were maden through the visual and programmatic assessments made.
# 
# 
# ### Quality Issues
# 
# #### twitter_arch dataframe
# 
# <font color=#0877cc> Observation One</font>
# - Columns doggo, floofer, pupper and puppo have None for missing values.
# 
# <font color=#0877cc> Observation Two</font>
# - The source column has html tag `&lta&gt` which has the source and can be extracted and covertd to categorical datatype.
# 
# <font color=#0877cc> Observation Three</font>
# - text column has the link for the tweets and ratings at the end which can be removed.
# 
# <font color=#0877cc> Observation Four</font>
# - timestamp column is str instead of datetime
# 
# <font color=#0877cc> Observation Five</font>
# - The rating_numerator column should of type float and also it should be correctly extracted.
# 
# <font color=#0877cc> Observation Six</font>
# - rating_denominator column has values less than 10 and values more than 10 for ratings more than one dog.
# 
# <font color=#0877cc> Observation Seven</font>
# - expanded_urls column has NaN values
# 
# <font color=#0877cc> Observation Eight</font>
# - name column have None instead of NaN and too many unvalid values.
# 
# <font color=#0877cc> Observation Nine</font>
# - id column in twitter_json_copy name different than the other 2 data sets.
# 
# 
# ### Tidiness Issues
# 
# twitter_arch dataframe
# <font color=#0877cc> Observation One</font>
# - doggo, floofer, pupper, puppo columns are all about the same things, that is dog stages.d
# 
# <font color=#0877cc> Observation Two</font>
# - img_num does not have any usage in the dataset.
# 
# <font color=#0877cc> Observation Three</font>
# - Just 3 columns needed id, retweet_count, favorite_count.
# 
# ### General
# <font color=#0877cc> Observation </font>
# - All datasets should be combined into 1 dataset only.

# # CLEANING

# First, copies of each dataframes need to be made before cleaning them.

# In[30]:


# Making copies of each dataset
twitter_arch_copy = twitter_arch.copy()
img_pred_copy = img_pred.copy()
twitter_json_copy = twitter_json.copy()


# Define
# Replace 'None' with np.nan for doggo, floofer, pupper and puppo columns.

# ### Code

# In[32]:


# Replacing all 'None' values in the `doggo`, `floofer`, `pupper` and `puppo` columns.
dog_category = ['doggo', 'floofer', 'pupper', 'puppo']

for dog in dog_category:
    twitter_arch_copy[dog] = twitter_arch_copy[dog].replace('None', np.nan)


# ### Test

# In[33]:


# Testing the sample of the replaced data
twitter_arch_copy.sample(5)


# In[34]:


# Checking the summary statistics of the `doggo`, `floofer`, `pupper` and `puppo` columns.
twitter_arch_copy[['doggo', 'floofer', 'pupper', 'puppo']].info()


# ### Define
# Extract tweet source from source column using 'apply method' in pandas and then convert it to categorical datatype

# ### Code

# In[35]:


# Checking for the unique values
twitter_arch_copy.source.unique()


# In[36]:


#Defining a function fix_source which extract the strings between tags
def fix_source(x):
    'x is an html string from the source column in df_arch_cleaned dataset'
    # Finding the first closed  tag >
    i= x.find('>') + 1
    # Finding the first open tag after the previous <
    j =x[i:].find('<')
    # Extracting the text in between
    return x[i:][:j]


# In[37]:


# Converting the `source` column to a category datatype
twitter_arch_copy.source = twitter_arch_copy.source.apply(lambda x: fix_source(x)).astype('category')


# Test

# In[38]:


# Checking for unique values in the `source` column
twitter_arch_copy.source.unique()


# Define
# Extract rating scores from tweet text using RegEx and convert it to float

# Code

# In[39]:


# Checking the `text` column for rows that contains the texts to be extracted
twitter_arch_copy[twitter_arch_copy.text.str.contains(r"(\d+\.\d*\/\d+)")]                                            [['text', 'rating_numerator']]


# In[40]:


# Extracting the ratings from the text column
new_ratings = (twitter_arch_copy[twitter_arch_copy.text.str.contains
              (r"(\d+\.\d*\/\d+)")]['text'].str.extract
              (r"(\d+\.\d*(?=\/\d+))"))
new_ratings


# In[41]:


# Assigning the new ratings values to the `rating_numberator` column
twitter_arch_copy.loc[new_ratings.index, 'rating_numerator'] = new_ratings.values


# In[42]:


# Converting the assigned values to float datatype
twitter_arch_copy.rating_numerator = twitter_arch_copy.rating_numerator.astype('float')


# ### Test

# In[43]:


twitter_arch_copy.loc[new_ratings.index]


# In[44]:


twitter_arch_copy.info()


# ### Define
# Convert the timestamp column to datetime.

# ### Code

# In[45]:


# COnverting `timestamp` column to datatime datatype
twitter_arch_copy.timestamp = pd.to_datetime(twitter_arch_copy.timestamp)


# Test

# In[46]:


# Comfirming the change of the `timestamp` datatype to datetime.
twitter_arch_copy.timestamp.dtype


# ### Define
# Remove values other than 10 for rating_denominator
# 
# 
# ### Code

# In[47]:


# Removing all rows in `rating_denominator` that are not equal to 10
twitter_arch_copy = twitter_arch_copy[twitter_arch_copy['rating_denominator'] == 10]


# Test

# In[48]:


# Describing the summary statistics of the `rating_numerator` 
# `rating_denominator` columns to comfirm the removal of all
# rows that are not having denominators of 10
twitter_arch_copy[['rating_numerator', 'rating_denominator']].describe()


# ### Define
# Remove any rows not related to dogs
# 
# ### Code

# In[50]:


# Checking for null values in the `text` column
twitter_arch_copy = twitter_arch_copy[~twitter_arch_copy.text.isnull()]


# In[51]:


# Removing all columns that are not related to "only rate dogs"
twitter_arch_copy = twitter_arch_copy.loc[~twitter_arch_copy.text.str.match('.*only rate dogs')]


# ### Test

# In[52]:


# Confirming for rows that are not related to "only rate dogs"
twitter_arch_copy.loc[twitter_arch_copy.text.str.match('.*only rate dogs')]


# In[53]:


# Getting a summary of the columns
twitter_arch_copy.info()


# ### Define
# Drop rows with NaNs values in the expanded_urls column.
# 
# 
# ### Code

# In[54]:


# Locating and droping all NaNs values in the `expanded_urls` column
twitter_arch_copy = twitter_arch_copy.loc[~twitter_arch_copy.expanded_urls.isnull()]


# In[55]:


# Confirming the romoval of all NaN values in the `expanded_urls` column
twitter_arch_copy.info()


# ### Define
# Create dog_stage column and remove the (doggo, floofer, pupper, puppo) columns.

# ### Code

# In[56]:


# Selecting the dog stages columns from the dataset
dog_category = ['doggo', 'floofer', 'pupper', 'puppo']

# Creating the dog_stage column by joining the four columns
twitter_arch_copy['dog_stage'] = (twitter_arch_copy[dog_category].
                                        apply(lambda x: ', '.join(x.dropna().astype(str)),axis =1))
# Replacing the empty string with NaN and changing the datatype to category
twitter_arch_copy.dog_stage = twitter_arch_copy.dog_stage.replace('', np.nan).astype('category')

# Dropping the 4 columns
twitter_arch_copy = twitter_arch_copy.drop(columns = dog_category, axis =1)


# ### Test

# In[57]:


# Confirming the merging of the four columns 'doggo', 'floofer', 'pupper' and  'puppo' to 'dog_stage'
twitter_arch_copy.info()


# In[58]:


# Checking the value count of the stages of dog in the `dog_stage` column
twitter_arch_copy.dog_stage.value_counts()


# ### Define
# Replace 'None' with np.nan in twitter_arch_copy name column.
# Remove any rows with invalid names which starts with lower laters.
# 
# ### Code

# In[59]:


# Identifying unique names that are are not names of dogs in the `name` column
twitter_arch_copy[~twitter_arch_copy.name.str.istitle()].name.unique()


# In[60]:


# Replacing the names that are not dogs name with NaN in the `name` column
twitter_arch_copy.name.replace(['such', 'a', 'quite', 'not', 'one', 'incredibly', 'mad',
       'an', 'very', 'just', 'my', 'his', 'actually', 'getting',
       'this', 'unacceptable', 'all', 'old', 'infuriating', 'the',
       'by', 'officially', 'life', 'light', 'space', 'None'], np.nan, inplace=True)


# ### Test

# In[61]:


# Checking for the unique names in the `name` column
twitter_arch_copy.name.unique()


# In[62]:


# Checking the value counts for each name in the `name` column
twitter_arch_copy.name.value_counts()


# In[63]:


# Checking the summary of the various columns
twitter_arch_copy.info()


# ### Define
# Remove img_num column from img_pred_copy dataset.
# 
# ### Code

# In[64]:


# Looking up for the columns 
img_pred_copy.columns


# In[65]:


# Removing the `img_num` column
img_pred_copy.drop('img_num', axis=1, inplace=True)


# ### Test

# In[66]:


# Confirming the removal of the `img_num` column
img_pred_copy.info()


# ### Define
# Create breed and confidence columns with highest confidence predictions and drop other columns
# 
# ### Code

# In[67]:


# Creating an empty list of `breed` and `confidence`
breed = []
confidence = []

# Iterating over img_pred_copy row by row and taking 
# the highest confident prediction otherwise np.nan
for index, row in img_pred_copy.iterrows():
    if row['p1_dog'] and row['p1_conf'] == max([row['p1_conf'], row['p2_conf'], row['p3_conf']]):
        breed.append(row['p1'])
        confidence.append(row['p1_conf'])
    elif row['p2_dog'] and row['p2_conf'] == max([row['p1_conf'], row['p2_conf'], row['p3_conf']]):
        breed.append(row['p2'])
        confidence.append(row['p2_conf'])
    elif row['p3_dog'] and row['p3_conf'] == max([row['p1_conf'], row['p2_conf'], row['p3_conf']]):
        breed.append(row['p3'])
        confidence.append(row['p3_conf'])
    else:
        breed.append(np.nan)
        confidence.append(np.nan)
        
img_pred_copy['breed'] = breed
img_pred_copy['confidence'] = confidence


# In[68]:


# Re-defining the new columns for the img_pred_copy dataset
img_pred_copy = img_pred_copy[['tweet_id', 'jpg_url', 'breed', 'confidence']]


# ### Test

# In[69]:


# Sampling the data to find out the change in columns
img_pred_copy.sample(5)


# In[70]:


# Confirming info on the columns
img_pred_copy.info()


# ### Define
# Remove unnecessary columns for twitter_json_copy
# 
# ### Code

# In[71]:


# Confirming the columns
twitter_json_copy.columns


# In[72]:


# Maintaining only `id`, `retweet_count` and `favorite_count` columns
twitter_json_copy = twitter_json_copy[['id','retweet_count', 'favorite_count' ]]


# ### Test

# In[73]:


# Confirming the columns in the twitter_json_copy dataset
twitter_json_copy.info()


# ### Define 
# Rename id column in twitter_json_copy to tweet_id
# 
# ### Code

# In[74]:


# Renaming the `id` column to `tweet_id`
twitter_json_copy.rename(columns = {'id':'tweet_id'}, inplace = True)


# ### Test

# In[75]:


# Confirming the changed column nname
twitter_json_copy.info()


# In[76]:


# Confirming the changed column nname
twitter_json_copy.columns


# ### Define
# Merge all the three cleaned datasets into one dataframe
# 
# ### Code

# In[77]:


# Merging the three datasets using the common column: `tweet_id`
df = pd.merge(twitter_arch_copy, img_pred_copy, on='tweet_id')
df = pd.merge(df, twitter_json_copy, on = 'tweet_id')


# ### Test

# In[78]:


# Getting the info of the columns of the new dataframe
df.info()


# In[79]:


# Checking the first five rows of the new dataframe
df.head()


# In[80]:


# Checking the last five rows of the new dataframe
df.tail()


# #### Cleaning the merged dataframe for further analysis and visualization

# In[81]:


# Converting the `breed` column to a category datatype
df.breed = df.breed.astype('category')


# In[82]:


# Testing for the changes
df.info()


# ## STORING

# In[83]:


# Saving the new dataframe as `twitter_archive_master.csv` 
df.to_csv('twitter_archive_master.csv', encoding = 'utf-8', index = False)


# In[84]:


# Reading back the save twitter_archive_master.csv file into a new dataframe
dog_rating_df = pd.read_csv('twitter_archive_master.csv')


# In[85]:


# Checking the first five rows of the newly created `dog_rating_df dataframe
dog_rating_df.head()


# ### VISUALIZATION
# 
# Considering the distribution of ratings using a bar graph

# In[86]:


# Ploting a bar graph for the to show the frequency of ratings
data = dog_rating_df.rating_numerator.value_counts()

x = data.index
y = data.values
fig, ax = plt.subplots(figsize=(12, 6))
g = sns.barplot(x, y, palette='Blues_d', ax=ax)
ax.set(xlabel='Ratings', ylabel='Frequency', title='Ratings Frequency')
plt.show()


# There seems to be some outlier in the ratings like 420.0 and 1776.0. This can be further investigated using a boxplot

# In[92]:


# Ploting a boxplot to identify the outliers in the frequency of ratings
data = dog_rating_df.rating_numerator.value_counts()

ax = sns.boxplot(y=data.values, data=data)
ax.set(xlabel='Ratings', ylabel='Frequency', title='Ratings frequency')
plt.show()


# From the boxplot, there are 2 outliers here which need further investigating and to check for their data

# In[89]:


# Checking out the outliers in the dataset
outliers_df = dog_rating_df[dog_rating_df.rating_numerator > 400][['rating_numerator', 'name', 'jpg_url', 'text']]
outliers_df


# To confirm why these dogs have the outlied ratings, a look at the dogs should be done by downloading their pictures from the given URL in the jpg_url column

# In[90]:


# Creating a folder called `images` to hold the images to be downloaded
if not os.path.exists('images'):
    os.makedirs('images')
    
# Downloading and displaying the pictures of the dogs
fig=plt.figure()
c = 1
for index, row in outliers_df.iterrows():
    r = requests.get(row['jpg_url'])
    i = Image.open(BytesIO(r.content))
    i.save('images/' +  str(index) + '_' + str(row['rating_numerator']) + "_" + str(row['name']) + '.jpg')
    fig.add_subplot(1, 2, c)
    c += 1
    plt.imshow(i)
    plt.axis("off")
plt.show()


# The first dog shown seem to deserve its rating as well as the American rapper Snoop Dogg. This seems to be the reason why they are outliers
# 
# #### Considering the relation between retweet_count and favorite_count

# In[108]:


# Plotting a Scatter plot to show the relation between favorits and retweets
sns.regplot(x=dog_rating_df["retweet_count"], y=dog_rating_df["favorite_count"], fit_reg=False)
ax.set(xlabel='Retweet count', ylabel='Favorite count', title='Favorits VS Retweets')
plt.show()


# There seems to be a positive correlation between 'retweet_count' and 'favorite_count'. A further look at this using a regplot could also be used to confirm this colleration

# In[93]:


# Plotting a regplot to show the relation between favorits and retweets
ax = sns.regplot(x='retweet_count', y='favorite_count', data=dog_rating_df, color='b', scatter_kws={'s':5, 'alpha':.3}) 
ax.set(xlabel='Retweet count', ylabel='Favorite count', title='Favorits VS Retweets')
plt.show()


# The regplot shows the positive correlation between 'retweet_count' and 'favorite_count'. Therefore, the most liked dogs were retweeted most, thus the higher the like, the more the retweet.
# 
# ### Comparing the numbers of the various stages of dogs
# There would be the need for a bar graph function to be used for the rest of the visualization. A plot_bar function can be used.

# In[109]:


# Defining a 'plot_bar' function.
def plot_bar(x, y, data, title = " ", xlabel = " ", ylabel = " ", rotation = 90):
    plt.figure(figsize = (10, 8))
    bar_list = sns.barplot(x=x, y=y,)
    plt.title(title, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    
    return plt.show()


# In[110]:


# Showing the value count for each dog stage
dog_stage_count = dog_rating_df.dog_stage.value_counts()
dog_stage_count


# 'Pupper' seems to have the highest value count. This can further be presented visually using a bar plot

# In[96]:


# Plotting a bar graph to show the numbers of the various satges of dogs
data = dog_rating_df.groupby('dog_stage').count()['tweet_id']

plot_bar(data.values, data.index, data, "Dog Stage Counts", "Count", "Dog stage");


# From the bar graph, 'pupper' is the dog stage that appeared most.
# 
# *A further consideration of which stages of dogs were retweeted more frequently could also be done**

# In[97]:


data = dog_rating_df.groupby('dog_stage').count()['retweet_count']
plot_bar(data.values, data.index, data, "Most retweeted Dog Stage", "Count", "Dog Stages");


# Just as the dog stage count, from the bar graph, 'pupper' is the most retweeted dog stage.
# 
# #### Comparing the various sources of the tweets

# In[98]:


# Showing the value count for each dog stage
tweet_source_count = dog_rating_df.source.value_counts()
tweet_source_count


# It seems most tweets were made from iPhones. This can further be presented visually using a bar plot

# In[99]:


# Plotting a bar graph to show the numbers of the various satges of dogs
data = dog_rating_df.groupby('source').count()['tweet_id']
#ax = sns.barplot(y=data.index, x=data.values, palette='Blues_d')
#ax.set(xlabel='Count', ylabel='Tweet source', title='Tweet Source Counts')
#plt.show()

plot_bar(data.values, data.index, data, "Tweet Source Counts", "Count", "Tweet source");


# From the bar graph it can be seen that the 'Twitter for iPhone' is the highest source of tweets
# 
# Top 12 most used dog names

# In[112]:


top12_dog_names= dog_rating_df.name.value_counts().head(12)
top12_dog_names


# Charlie appears as the most used dog name, followed by Lucy, Penny and the others. This can further be presented visually using a bar plot

# In[113]:


# Plotting a bar graph to show the top 12 most used dog names
data = top12_dog_names
plot_bar(data.values, data.index, data, "Top 12 Most Used Dog Names", "Count", "Dog Names");


# The bar graph shows that Charlie, Lucy, Penny, Oliver, Tucker, Cooper, Bo, Sadie, Winston, Lola, Toby and Daisy are the top 12 most used dog names. This can further be presented visually using a bar plot

# 
