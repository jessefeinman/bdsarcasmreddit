"""
get_tweets.py

usage: get_tweets.py [-h] [--sarcastic_path SARCASTIC_PATH]
                     [--non_sarcastic_path NON_SARCASTIC_PATH]
                     [--log_path LOG_PATH]

Query twitter API for tweets over last 7 days

optional arguments:
  -h, --help            show this help message and exit
  --sarcastic_path SARCASTIC_PATH
                        path to directory where results w/ #sarcasm should be
                        saved. Needs trailing "/"
  --non_sarcastic_path NON_SARCASTIC_PATH
                        path to directory where results w/o #sarcasm should be
                        saved. Needs trailing "/"
  --log_path LOG_PATH   path to save log. Needs trailing "/"
"""

import os
import sys
import json
import argparse
import logging

from datetime import datetime, timedelta

from TwitterSearch import *
from login import *
from json_io import list_to_json

def create_sarcastic_search_order():
    tso = TwitterSearchOrder()
    tso.set_keywords(['#sarcasm']) # query only tweets containing #sarcasm
    tso.set_language('en')
    tso.set_include_entities(True)
    tso.arguments.update({"tweet_mode": "extended"})
    return tso

def create_non_sarcastic_search_order():
    tso = TwitterSearchOrder()
    tso.set_keywords(["-#sarcasm"]) # must have keyword, so query tweets containing common words but NOT '#sarcasm'
    tso.set_language('en')
    tso.set_include_entities(True)
    tso.arguments.update({"tweet_mode": "extended"})
    return tso

if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(description='Query twitter API for tweets over last 7 days')
    parser.add_argument('--sarcastic_path', help='path to directory where results w/ #sarcasm should be saved. Needs trailing "/"')
    parser.add_argument('--non_sarcastic_path', help='path to directory where results w/o #sarcasm should be saved. Needs trailing "/"')
    parser.add_argument('--log_path', help='path to save log. Needs trailing "/"')

    # Parse CLAs
    args = parser.parse_args()

    # start and end date (for file naming/logging)
    end_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
    start_date =  datetime.strftime( (datetime.now() - timedelta(days=7)), "%Y-%m-%d")
    filename = "{}_{}".format(start_date, end_date)

    # setup logger
    if args.log_path:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        logger = logging.getLogger('root')
        FORMAT = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(filename=args.log_path + filename + ".log", filemode='a', level=logging.INFO, format=FORMAT)

    # lists to store tweets
    sarcastic_tweets_list = []
    non_sarcastic_tweets_list = []

    # create search orders
    if args.sarcastic_path:
        sarcastic_tso = create_sarcastic_search_order()
    if args.non_sarcastic_path:
        non_sarcastic_tso = create_non_sarcastic_search_order()

    try:
        # query twitter API and populate tweet lists
        ts = TwitterSearch(
            consumer_key = CONSUMER_KEY,
            consumer_secret = CONSUMER_SECRET,
            access_token = ACCESS_TOKEN,
            access_token_secret = ACCESS_SECRET
         )
        if args.sarcastic_path:
            for sarcastic_tweet in ts.search_tweets_iterable(sarcastic_tso):
                if not sarcastic_tweet['full_text'].lower().startswith('rt'):
                    sarcastic_tweets_list.append({
                        'id': sarcastic_tweet['id'],
                        'urls': not not sarcastic_tweet['entities']['urls'],
                        'media': "media" in sarcastic_tweet["entities"],
                        'text': sarcastic_tweet['full_text']})
        if args.non_sarcastic_path:
            for non_sarcastic_tweet in ts.search_tweets_iterable(non_sarcastic_tso):
                if not non_sarcastic_tweet['full_text'].lower().startswith('rt'):
                    non_sarcastic_tweets_list.append({
                        'id': non_sarcastic_tweet['id'],
                        'urls': not not non_sarcastic_tweet['entities']['urls'],
                        'media': "media" in non_sarcastic_tweet["entities"],
                        'text': non_sarcastic_tweet['full_text']})
    except TwitterSearchException as e:
        logging.error(str(e))

    # save results to json
    if args.sarcastic_path:
        if not os.path.exists(args.sarcastic_path):
            os.makedirs(args.sarcastic_path)
        list_to_json(sarcastic_tweets_list, args.sarcastic_path + filename + ".json")
        if args.log_path:
            logging.info("Saved {} sarcastic tweets at {}".format(len(sarcastic_tweets_list), args.sarcastic_path + filename + ".json"))
    if args.non_sarcastic_path:
        if not os.path.exists(args.non_sarcastic_path):
            os.makedirs(args.non_sarcastic_path)
        list_to_json(non_sarcastic_tweets_list, args.non_sarcastic_path + filename + ".json")
        if args.log_path:
            logging.info("Saved {} non sarcastic tweets at {}".format(len(non_sarcastic_tweets_list), args.non_sarcastic_path + filename + ".json"))
