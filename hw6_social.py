"""
Social Media Analytics Project
Name:Sneha.K
Roll Number:2021501022
"""

from pandas.core import indexing
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import re
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
df={}
'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    return pd.read_csv(filename)

    


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
import re

def parseName(fromString): 
    name = re.findall("From:\s*(.*?)\s*\(", fromString) 
    if len(name)>0:
        return name[0]
    else: 
        return ''



'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    position = re.findall("\s\((.*?)\sfrom",fromString)
    # print(position[0],'\n')
    if len(position)>0:
        return position[0]
    else: 
        return ''

'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    state = re.findall(".*from\s(.*?)\)",fromString)
    if len(state)>0:
        return state[0]
    else: 
        return ''

'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    # taglist= re.findall("#\w+",message)
    # print("\n",taglist,"\n")
    tags=[]
    msglst=message.split('#')
    templst=[]
    for each in msglst[1:]:
        strn=""
        for char in each:
            if char in endChars:
                break
            strn=strn+char
        strn="#"+strn
        tags.append(strn)
    return tags

    # return taglist

'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    regionresult = stateDf.loc[stateDf['state'] == state, 'region']
    
    return regionresult.values[0]

'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index,row in data.iterrows():
        stringinrow = row["label"]
        names.append(parseName(stringinrow))
        positions.append(parsePosition(stringinrow))
        stateinrow=parseState(stringinrow)
        states.append(stateinrow)
        regions.append(getRegionFromState(stateDf,stateinrow))
        text=row["text"]
        hashtags.append(findHashtags(text))
    data["name"]=names
    data['position']=positions
    data['state']=states
    data['region']=regions
    data['hashtags']=hashtags
    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index,row in data.iterrows():
        senti= findSentiment(classifier,row["text"])
        sentiments.append(senti)
    data["sentiment"]=sentiments
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    dict_count={}
    # print(data["state"])
    if dataToCount=="" and colName=="":
        for index,row in data.iterrows():
            if row["state"] not in dict_count:
                dict_count[row["state"]] = 1
            else:
                dict_count[row["state"]]+=1
    else:
        for index,row in data.iterrows():
            if dataToCount == row[colName] :
                if row["state"] not in dict_count:
                    dict_count[row["state"]] = 1
                else:
                    dict_count[row["state"]]+=1
    return dict_count


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    nested_dict={}
    for index,row in data.iterrows():
        nested_dict[row["region"]]={}
    for index,row in data.iterrows():
        if row[colName] not in nested_dict[row["region"]]:
            nested_dict[row["region"]][row[colName]]=1
        else:
            nested_dict[row["region"]][row[colName]]+=1
    return nested_dict


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    dict_hashtags={}
    for index,row in data.iterrows():
        for each in row["hashtags"]:
            if each not in dict_hashtags.keys():
                dict_hashtags[each]=1
            else:
                dict_hashtags[each]+=1
    # print(len(dict_hashtags))
    # print(dict_hashtags)
    return dict_hashtags


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    common_hashtags={}
    Most_common_hashtag={}
    common_hashtags=sorted(hashtags.items(),key= lambda x:x[1],reverse=True)
    for each in common_hashtags[0:count]:
        Most_common_hashtag[each[0]]=each[1]
    return Most_common_hashtag



'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    hashtag_list=[]
    count=0
    all_sentiments=[]
    for index,row in data.iterrows():
        # print(row["sentiment"])
        hashtag_list=findHashtags(row["text"])
        if hashtag in hashtag_list:
            count+=1
            hashtag_sentiment= row["sentiment"]
            if hashtag_sentiment == "positive" : all_sentiments.append(1)
            elif hashtag_sentiment == "negative" : all_sentiments.append(-1)
            else : all_sentiments.append(0)
    avg_hashtag_sentimeent= sum(all_sentiments)/count

    return avg_hashtag_sentimeent

### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()

    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""
    # test.testAddColumns()
    # test.testFindSentiment()
    # test.testAddColumns()
    # # test.testParseName()
    # # test.testParsePosition()
    # test.testParseState()
    df = makeDataFrame("data/politicaldata.csv")
    stateDf = makeDataFrame("data/statemappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    test.testGetHashtagRates(df)
