"""
process_tweets.py

Template for processing tweets
"""


import glob
import argparse

from json_io import list_to_json, list_from_json, tweet_map


if __name__ == "__main__":
    # setup CLA parser
    parser = argparse.ArgumentParser(description='Process tweets')
    parser.add_argument('path', help='path to tweet json to process on. Needs trailing "/"')
    # parse CLAs
    args = parser.parse_args()
    JSON_PATH = args.path
    # example function to apply to each tweet
    def tweet_func(tweet):
        tweet["test"] = True
        return tweet
    # process tweets, optionally overwrite with save parameter
    tweet_map(JSON_PATH, tweet_func, save=False)
