"""
get_unique.py

USAGE: get_unique.py [-h] [--sarcastic_path SARCASTIC_PATH]
                     [--non_sarcastic_path NON_SARCASTIC_PATH]

Create one json file with unique tweets

optional arguments:
  -h, --help            show this help message and exit
  --sarcastic_path
                        path to directory of sarcastic tweet jsons. Needs
                        trailing "/"
  --non_sarcastic_path
                        path to directory of non sarcastic tweet jsons. Needs
                        trailing "/"
"""

import glob
import json
import os
import argparse

from json_io import list_to_json, list_from_json

if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(description='Create one json file with unique tweets')
    parser.add_argument('--sarcastic_path', help='path to directory of sarcastic tweet jsons. Needs trailing "/"')
    parser.add_argument('--non_sarcastic_path', help='path to directory of non sarcastic tweet jsons. Needs trailing "/"')

    # Parse CLAs
    args = parser.parse_args()
    top_lvl_paths_lst = []
    if args.sarcastic_path:
        if not os.path.exists(args.sarcastic_path):
            raise Exception("Invalid path: {}".format(args.sarcastic_path))
        top_lvl_paths_lst.append(args.sarcastic_path)
    if args.non_sarcastic_path:
        if not os.path.exists(args.non_sarcastic_path):
            raise Exception("Invalid path: {}".format(args.non_sarcastic_path))
        top_lvl_paths_lst.append(args.non_sarcastic_path)

    # set static filenames
    FN_HASH = "hash_dict.json"
    FN_UNIQUE = "unique.json"

    # Populate list with paths to jsons
    json_paths_lst = [glob.glob(p + "*-*-*_*-*-*.json") for p in top_lvl_paths_lst]

    # Find and save unique tweets and updated hash dict
    for json_paths, top_lvl_path in zip(json_paths_lst, top_lvl_paths_lst):
        # load in existing list of unique tweets if it exists
        unique_tweets_lst = []
        if os.path.exists(top_lvl_path + FN_UNIQUE):
            unique_tweets_lst = list_from_json(top_lvl_path + FN_UNIQUE)
        # load in existing hash dict if it exists
        hash_dict = {}
        if os.path.exists(top_lvl_path + FN_HASH):
            hash_dict = list_from_json(top_lvl_path + FN_HASH)

        # populate list with all tweets (possibly non-unique) for user passed directory
        tweets_lst = [ tweet for json_path in json_paths for tweet in list_from_json(json_path)]
        # for each tweet, check its existence in hash dict. Update unique list and hash dict
        for tweet in tweets_lst:
            if str(tweet['id']) not in hash_dict:
                unique_tweets_lst.append(tweet)
                hash_dict[str(tweet['id'])] = True
        # Save updated unique tweets list and hash dict
        list_to_json(unique_tweets_lst, top_lvl_path + FN_UNIQUE)
        list_to_json(hash_dict, top_lvl_path + FN_HASH)
