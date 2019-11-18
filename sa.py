
import sklearn
import sklearn.naive_bayes
import sklearn.model_selection
import numpy as np
import csv
import string
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import f1_score

# Load the data file
def loadData(filename):
    data = open(filename,encoding='utf-8-sig')
    i=0
    columns = {}
    review_text = []
    rating = []
    recommend = []
    for row in csv.reader(data, delimiter=','):
        if (i == 0):
            # Build column index
            for j in range(0, len(row)):
                columns[row[j]] = j
        else:
            # Store relevant data
            if row[columns["reviews.rating"]] != '' and row[columns["reviews.doRecommend"]] != '':
                review_text.append(row[columns["reviews.text"]])
                rating.append(row[columns["reviews.rating"]])
                recommend.append(row[columns["reviews.doRecommend"]])
        i += 1
    return [review_text, rating, recommend]

# Convert to lowercase, remove punctuation
def cleanText(data_string):
    data_string = data_string.translate(str.maketrans('', '', string.punctuation))
    data_string = data_string.lower()
    return data_string

# Remove stopwords
def filterStopWordsTags(tokens):
    return [word for word in tokens if not word[0] in stopwords.words('english')]

def filterStopWords(tokens):
    return [word for word in tokens if not word in stopwords.words('english')]

# Get only the tags
def tagsOnly(tokens):
    return [x[1] for x in tokens]
    
def tagsToString(tokens):
    return ' '.join( ['{}/{}'.format(x[0], x[1]) for x in tokens] )

def chunktree(tagged):
    chunkGram = r"""Chunk: {<RB.?><RB.?>*<VB.?><VB.?>*}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    chunked.draw()


# Load the training data from file
print("Loading...")
review_text,ratings,recommend = loadData('consumer-reviews-of-amazon-products/1429_1.csv')
Xtrain = review_text

ratings = [int(x) for x in ratings]

# Clean the data
print("Cleaning...")
Xtrain = [cleanText(x) for x in Xtrain]

# Split into tokens
print("Splitting...")
Xtrain = [x.split() for x in Xtrain]

# POS Tags
print("Tagging...")
Xtrain = [nltk.pos_tag(x) for x in Xtrain]

# Remove stopwords
print("Filtering...")
#Xtrain = [filterStopWords(x) for x in Xtrain]
Xtrain = [filterStopWordsTags(x) for x in Xtrain]
#Xtrain = [tagsOnly(x) for x in Xtrain]

# Rebuild string
print("Concatenating...")
Xtrain = [tagsToString(x) for x in Xtrain]
#Xtrain = [' '.join(x) for x in Xtrain]

# Create feature vector based on word counts
print("Vectorizing...")
vectorizer = sklearn.feature_extraction.text.CountVectorizer()
#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
Xtrain_count_vectors = vectorizer.fit_transform(Xtrain).toarray()

# Test with cross validation
print("Testing...")
rating_predict_accuracy = 0.0
recommend_predict_accuracy = 0.0
rating_predict_fscore = 0.0
recommend_predict_fscore = 0.0

k = 5
kfold = sklearn.model_selection.KFold(k, True)
for train, test in kfold.split(Xtrain_count_vectors):
    # train set
    train_set = [Xtrain_count_vectors[i] for i in train]
    train_ratings = [ratings[i] for i in train]
    train_reccomend = [recommend[i] for i in train]

    # test set
    test_set = [Xtrain_count_vectors[i] for i in test]
    test_ratings = [ratings[i] for i in test]
    test_reccomend = [recommend[i] for i in test]

    # predict ratings
    model = sklearn.naive_bayes.MultinomialNB().fit(train_set, train_ratings)
    predictions = model.predict(test_set)
    rating_predict_accuracy += np.mean(predictions == test_ratings)
    rating_predict_fscore += f1_score(predictions, test_ratings, average='micro')
    
    # predict recommendation
    model = sklearn.naive_bayes.MultinomialNB().fit(train_set, train_reccomend)
    predictions = model.predict(test_set)
    recommend_predict_accuracy += np.mean(predictions == test_reccomend)
    recommend_predict_fscore += f1_score(predictions, test_reccomend, average='micro')
    
# print results
print("Rating Predict Accuracy:\t", float(rating_predict_accuracy)/float(k))
print("Rating Predict F-Score:\t\t", float(rating_predict_fscore)/float(k))

print("\nRecommend Predict Accuracy:\t", float(recommend_predict_accuracy)/float(k))
print("Recommend Predict F-Score:\t", float(recommend_predict_fscore)/float(k))
