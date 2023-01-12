import json, pickle

def loadJson(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def dumpPickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)