"""
json_io.py

Functions related to reading/writing/mapping json
"""

import json
import ijson
from re import sub
from datetime import datetime
from os import listdir, SEEK_END
from sys import exc_info, stdout
from random import choice, randrange
from nlp import feature

TWEET_LINK_RE = "https://t.co/(\w)+"
TWEET_HANDLE_RE = "@(\w)+"

def list_from_json(json_file):
    """Return a list corresponding to contents of json file"""

    with open(json_file, 'r') as fp:
        return json.load(fp)

def list_to_json(lst, path, old_format=True):
    """Save a list of tweets to a json file at corresponding path.
    old_format (optional, default=true): dump using sorted keys, indenting. Set to false for streaming friendlier format
    """

    if old_format:
        with open(path, 'w') as fp:
            json.dump(lst, fp, sort_keys=True, indent=4)
    else:
        with open(path, 'w') as fp:
            for i, tweet in enumerate(lst):
                json.dump({"text": tweet["text"], "id": tweet['id'], "media": tweet["media"], "urls": tweet["urls"]}, fp)
                if i != len(lst) - 1:
                    fp.write('\n')

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

def fileName(features_path, source, sarcastic, i=None):
    return features_path + source + ('sarcastic-' if sarcastic else 'serious-') + str(i) + ".json"

def openFiles(features_path, sarcastic, source, n, mode='a'):
    """
    takes in a directory path, a sarcastic boolean value, a source type and n
    Returns n file pointers in the specified (defaul append) mode with a large buffer located in the feature_path directory.
    feature_path= feats
    sarcastic = True
    source = tweet-
    n=5
    Will create files like so:
    feats/tweet-sarcastic-0.json
    feats/tweet-sarcastic-1.json
    ...
    feats/tweet-sarcastic-5.json
    """
    return [open(fileName(features_path, source, sarcastic, i), mode, buffering=2**24) for i in range(n)]

def closeFiles(openFiles):
    """
    Takes in a list of open file pointers
    flushes the buffer (done in file.close()) and closes the files.
    """
    for file in openFiles:
        file.close()

def processRandomizeJson(sarcastic, json_path, features_path, source, n, cleanTokens):
    """
    takes in a sarcastic boolean, a path to json files, a path to store processed features, a source type an the number of files to create
    For each json file in the json_path directory it processes the features and saves it randomly to 1 of n files constructed using the openFiles function
    Periodically prints the file and time it took to process as well as the number of items processed so far.
    """
    files = openFiles(features_path, sarcastic, source, n, mode='a')
    try:
        totalCount = 0
        for filename in listdir(json_path):
            startTime = datetime.now()
            for line in open(json_path+filename):
                text = json.loads(line)['text']
                features = feature(text, cleanTokens)
                featuresJson = json.dumps(features) + '\n'
                choice(files).write(featuresJson)
                totalCount += 1
            stopTime = datetime.now()
            print("File %s\ttime:\t%s" % (filename, (stopTime - startTime)))
            print("Processed %d json lines"%totalCount)
            stdout.flush()
        closeFiles(files)
    except:
        closeFiles(files)
        print("Unexpected error:\n")
        for e in exc_info():
              print(e)

def loadProcessedFeatures(features_path, source, sarcastic, n=0, feature_filename=None, random=True):
    if feature_filename:
        with open(feature_path+feature_filename) as file:
            for line in file:
                yield (json.loads(line), sarcastic)
    elif random:
        with open(fileName(features_path, source, sarcastic, randrange(n))) as file:
            for line in file:
                yield (json.loads(line), sarcastic)
    else:
        files = openFiles(features_path, sarcastic, source, n, mode='r')
        for file in files:
            for line in file:
                yield (json.loads(line), sarcastic)
