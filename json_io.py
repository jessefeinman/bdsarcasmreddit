"""
json_io.py

Functions related to reading/writing/mapping json
"""

import json
import ijson
from re import sub
from datetime import datetime

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

def json_map(json_file, json_func, save=False):
    """
    Apply a function to each tweet in a json file

    json_file - path to tweet json file
    tweet_func - function that takes in a 'tweet' object, and returns a 'tweet' object
    save (optional) - overwrite json_file with modified json

    returns list where each json has json_func applied to it
    """

    mapped_jsons = []
    with open(json_file, 'r') as f:
        # stream through f using ijson.items
        for json in ijson.items(f, "item"):
            mapped_jsons.append(json_func(json))
    if save:
        list_to_json(mapped_jsons, json_file)
    return mapped_jsons

def json_map(jsons, json_func):
    """
    Apply a function to each json in a list of jsons
    """

    return [json_func(json) for json in jsons]

def json_iterate(json_file, key=None):
    """
    Stream through objects in a json file

    json_file - path to json json file
    key (optional) - single key value of interest (ex: return only "text" field, or only "id" field of each json)
    """

    with open(json_file, 'r') as f:
        if key:
            for json in ijson.items(f, "item.{}".format(key)):
                yield json
        else:
            for json in ijson.items(f, "item"):
                yield json