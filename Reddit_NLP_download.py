''' 
Code to download and analyze reddit data. If you use it be kind!
Could certainly be much improved. 
'''

from __future__ import print_function
import praw
import pprint
import nltk
import bs4
import codecs
import os

# Some code below is thanks to (https://medium.com/@pasdan/how-to-scrap-reddit-using-pushshift-io-via-python-a3ebcc9b83f4)
import math
import json
import requests
import itertools
import numpy as np
import time
from datetime import datetime, timedelta

# Which subreddit?
subreddit = 'Liberal'

def make_request(uri, max_retries = 5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        return json.loads(response.content)    
	
    current_tries = 1

    while current_tries < max_retries:
        try:
            time.sleep(1)
            response = fire_away(uri)
            return response
        except:
            time.sleep(1)
            current_tries += 1    
			
    return fire_away(uri)

def pull_posts_for(subreddit, start_at, end_at):
    
    def map_posts(posts):
        return list(map(lambda post: {
            'id': post['id'],
            'created_utc': post['created_utc'],
            'prefix': 't4_'
        }, posts))
    
    SIZE = 500
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}'
    
    post_collections = map_posts( \
        make_request( \
            URI_TEMPLATE.format( \
                subreddit, start_at, end_at, SIZE))['data'])    
				
    n = len(post_collections)
    while n == SIZE:
        last = post_collections[-1]
        new_start_at = last['created_utc'] - (10)
        
        more_posts = map_posts( \
            make_request( \
                URI_TEMPLATE.format( \
                    subreddit, new_start_at, end_at, SIZE))['data'])
        
        n = len(more_posts)
        post_collections.extend(more_posts)
    return post_collections

def give_me_intervals(start_at, number_of_days_per_interval = 3):
    
    end_at = math.ceil(datetime.utcnow().timestamp())
        
    ## 1 day = 86400,
    period = (86400 * number_of_days_per_interval)
    end = start_at + period
    yield (int(start_at), int(end))    
    
    padding = 1
    while end <= end_at:
        start_at = end + padding
        end = (start_at - padding) + period
        yield int(start_at), int(end)

# Start download

common_startdate = datetime(2020, 5, 23, 14, 52, 39, 944790)
start_at = math.floor(\
    ( common_startdate - timedelta(days=1095)).timestamp()) # formerly datetime.utcnow()

posts = []
for interval in give_me_intervals(start_at, 7):
    pulled_posts = pull_posts_for(
        subreddit, interval[0], interval[1])
    
    posts.extend(pulled_posts)
    print(posts[len(posts)-1])
    time.sleep(.500)
print(len(posts))
print(len(np.unique([ post['id'] for post in posts ])))


# The following uses my reddit ID to make an instance
# Each user must have their own credentials

reddit = praw.Reddit(client_id='notmyclientid',
                     client_secret='notmyclientsecret',
                     user_agent='notmyuseragent')

subreddit = reddit.subreddit(subreddit)

## WARNING: REDDIT WILL THROTTLE YOU IF YOU ARE ANNOYING! BE KIND!
TIMEOUT_AFTER_COMMENT_IN_SECS = .350
posts_from_reddit = []
comments_from_reddit = []
titles = []
texts = []

for submission_id in np.unique([ post['id'] for post in posts ]):
    submission = reddit.submission(id=submission_id)
    print('**********')
    print('Title is: ' + submission.title + '\n\n')    
    posts_from_reddit.append(submission)
    
    current_filename = submission.title
    current_filename = current_filename.replace("/","[slash]")
    current_filename = current_filename.replace("?","[questionmark]")
    if len(current_filename) > 50:
        current_filename = current_filename[:50]
    json_filename = current_filename + '.json'
    current_filename = current_filename + '.utf8'
    titles.append(submission.title)
    
    print('Description is: ' + submission.selftext + '\n\n')
    soup = bs4.BeautifulSoup(submission.selftext)
    current_text = bs4.BeautifulSoup.get_text(soup)
    texts.append(submission.selftext)

    try:
        if not os.path.exists(current_filename):
            output_file = codecs.open(current_filename, 'w+','utf-8')
            #print(submission.selftext, file=output_file)
            #if not os.path.isfile(current_filename):
            output_file.write(current_text)
            output_file.close()
    except:
        pass

    # Submission data to be written
    current_data = {
            "id" : submission_id,
            "title" : submission.title,
            "upvote_ratio" : submission.upvote_ratio,
            "num_comments" : submission.num_comments,
            "score": submission.score
        }
    try: 
        if not os.path.exists(json_filename):
            with open(json_filename, "w") as outfile:
                json.dump(current_data, outfile)
    except:
        pass

    submission.comments.replace_more(limit=None)

    comment_count = 0
    for comment in submission.comments.list():
        comments_from_reddit.append(comment)
        comment_json_filename = current_filename[:-5] + str(comment_count) + '.json'
        comment_filename = current_filename[:-5] + str(comment_count) + '.utf8'
           
        soup_comment = bs4.BeautifulSoup(comment.body)
        comment_text = bs4.BeautifulSoup.get_text(soup_comment)
        print('Comment is: ' + comment_text + '\n\n')
        try:
            if not os.path.exists(comment_filename):
                output_comment_file = codecs.open(comment_filename, 'w+','utf-8')
                output_comment_file.write(comment_text)
                output_comment_file.close()
                comment_count+=1
        except:
            pass
        
        # Submission data to be written
        current_comment_data ={
            "id" : comment.id,
            "title" : "",
            "upvote_ratio" : "",
            "num_comments" : "",
            "score": comment.score
        }
        try:
            if not os.path.exists(comment_json_filename):
                with open(comment_json_filename, "w") as outfile:
                    json.dump(current_comment_data, outfile)
        except:
            pass

        if TIMEOUT_AFTER_COMMENT_IN_SECS > 0:
            time.sleep(TIMEOUT_AFTER_COMMENT_IN_SECS)

print(len(posts_from_reddit))
print(len(comments_from_reddit))


