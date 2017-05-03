"""
squash_tweets.py

USAGE: squash_tweets.py [-h] [--delete] path

Squash multiple jsons into one

positional arguments:
  path        path to directory containing tweet jsons. Needs trailing "/"

optional arguments:
  -h, --help  show this help message and exit
  --delete    delete old JSONs after squashing
"""

import sys
import os
import glob
import argparse

from json_io import list_to_json, list_from_json, merge_json_filenames


if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(description='Squash multiple jsons into one')
    parser.add_argument('path', help='path to directory containing tweet jsons. Needs trailing "/"')
    parser.add_argument('--delete', action="store_true", help='delete old JSONs after squashing')
    # Parse CLAs
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise Exception("{} is not a valid path.".format(args.path))

    # Populate list with paths to jsons
    json_paths_lst = glob.glob(args.path + "*-*-*_*-*-*.json")

    # create one list of all merged tweets
    merged_lst = [ tweet for json_path in json_paths_lst for tweet in list_from_json(json_path)]

    # Get earliest and latest date of jsons squashed for naming purposes of merged file.
    filename = merge_json_filenames(json_paths_lst)

    # Save merged list to json
    list_to_json(merged_lst, args.path + filename)

    # Delete old files if flag is set
    if args.delete:
        for json_path in json_paths_lst:
            os.remove(json_path)
