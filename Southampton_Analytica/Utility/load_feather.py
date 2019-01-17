#Loads a feather file in and stores it as a dataframe object

import pandas as pd
import feather

#Specify the path to the feather. Will print all reviews in the text field.
def main():
    path = 'C:/example/data.feather'
    featherData = loadFeather(path)
    print(featherData['text'])

def loadFeather(path):
    return feather.read_dataframe(path)

main()
