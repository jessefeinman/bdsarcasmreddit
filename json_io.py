"""
json_io.py

Functions related to reading/writing/mapping json
"""

import json
import ijson
from re import sub
from datetime import datetime

TWEET_LINK_RE = "https://t.co/(\w)+"
TWEET_HANDLE_RE = "@(\w)+"

def list_from_json(json_file):
    """Return a list corresponding to contents of json file"""

    with open(json_file, 'r') as fp:
        return json.load(fp)

def list_to_json(lst, path):
    """Save a list to a json file at corresponding path."""

    with open(path, 'w') as fp:
        json.dump(lst, fp, sort_keys=True, indent=4)

def merge_json_filenames(json_lst):
    """
    Return filename encapsulating date range of passed in jsons
    ex: merge_json_filnames(["path/to/jsons/2017-01-27_2017-02-04.json", "path/to/jsons/2017-02-02_2017-02-09.json"])
        returns "2017-01-27_2017-02-09.json"
    """
    # Get earliest and latest date of jsons for naming purposes of merged file.
    parse_date_from_filename = lambda fn: fn.split('/')[-1].split('.')[0].split('_')
    sorted_dates = sorted([datetime.strptime(date, "%Y-%m-%d") for fn in json_lst for date in parse_date_from_filename(fn)])
    from_date = datetime.strftime(sorted_dates[0], "%Y-%m-%d")
    to_date = datetime.strftime(sorted_dates[-1], "%Y-%m-%d")
    return "{}_{}.json".format(from_date, to_date)

def tweet_map(json_file, tweet_func, save=False):
    """
    Apply a function to each tweet in a json file

    json_file - path to tweet json file
    tweet_func - function that takes in a 'tweet' object, and returns a 'tweet' object
    save (optional) - overwrite json_file with modified json

    returns list where each tweet has tweet_func applied to it
    """

    mapped_tweets = []
    with open(json_file, 'r') as f:
        # stream through f using ijson.items
        for tweet in ijson.items(f, "item"):
            mapped_tweets.append(tweet_func(tweet))
    if save:
        list_to_json(mapped_tweets, json_file)
    return mapped_tweets

def tweet_map(tweets, tweet_func):
    """
    Apply a function to each tweet in a list of tweets
    """

    return [tweet_func(tweet) for tweet in tweets]

def tweet_iterate(json_file, key=None):
    """
    Stream through objects in a json file

    json_file - path to tweet json file
    key (optional) - single key value of interest (ex: return only "text" field, or only "id" field of each tweet)
    """

    with open(json_file, 'r') as f:
        if key:
            for tweet in ijson.items(f, "item.{}".format(key)):
                yield tweet
        else:
            for tweet in ijson.items(f, "item"):
                yield tweet

def replaceLinksMentions(tweet):
    """
    Take tweet and return tweet with new field "ner_text" where links and handles are replaced by tokens
    """
    # replace embedded urls/media with [url], [media], or [url_media]
    ner_text = tweet["text"]
    if tweet["media"] or tweet["urls"]:
        if tweet['media'] and tweet['urls']:
            replacement_word = 'UrlMediaTOK'
        elif tweet['media']:
            replacement_word = "MediaTOK"
        else:
            replacement_word = "UrlTok"
        # replace twitter links with appropriate tag
        ner_text = sub(TWEET_LINK_RE, replacement_word, ner_text)
    # replace handles with appropriate tag
    ner_text = sub(TWEET_HANDLE_RE, "NameTOK", ner_text)
    tweet["ner_text"] = ner_text
    return tweet
