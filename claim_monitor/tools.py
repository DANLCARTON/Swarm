import re
import ml_upd_db as mud

def printTweet(text):
    print(text[:60]+"...", end = " ")

def keywordIterations(keyword ,tweet):
    tweet.lower()
    keyword.lower()
    iter = re.findall(keyword, tweet)
    return len(iter)

def searchKeywords(keywords, tweets):
    keywords.sort()
    for keyword in keywords:
        iter = 0
        for tweet in tweets:
            iter += keywordIterations(keyword, tweet)
        if iter != 0
            print(keyword, "appears", iter, "time(s)")
    print("all the keywords that are not mentioned are those that not appear in any tweet.")

def categorizeTweets(tweets):
    for tweet in tweets:
        printTweet(tweet)
        mud.testText(tweet)