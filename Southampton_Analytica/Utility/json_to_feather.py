#Convert source json into a feather file for easy access.

import pandas as pd
import numpy as np
import feather
import json

#Specify the input json file, and the output feather file location.
def main():
    input = 'C:/example/input.json'
    output = 'C:/example/output.feather'
    writeFeather(frameBuilder(input), output)
    featherData = loadFeather(output)
    print(featherData['text'])

#Loads each json container into an array, then adds them to a new dataframe.
def frameBuilder(path):
    reviews = []
    for row in open(path, 'r', encoding='UTF-8'):
        reviews.append(json.loads(row))
    return pd.DataFrame(reviews)

#Stores the dataframe as a feather in the output location.
def writeFeather(frame, path):
    frame.to_feather(path)

#Loads the feather back into a dataframe.
def loadFeather(path):
    return feather.read_dataframe(path)

main()
