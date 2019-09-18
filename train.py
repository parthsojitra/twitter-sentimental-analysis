# import all needed packages
import nltk
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

# create a class which extends ClassifierI Class
# this class use all classifiers classification and average them
class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	# create classify method which return maximum occured vote
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	# create confidence method which return confidence of that vote
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf
		
# read csv files for training
short_pos = pd.read_csv("processedPositive.csv")
short_neg = pd.read_csv("processedNegative.csv")

# read txt files for training
short_pos2 = open("positive_reviews.txt", "r").read()
short_neg2 = open("negative_reviews.txt", "r").read()

# create list for give label to training data 
docs =[]

# create strings for tokenize positive and negative reviews
s_pos = " "
s_neg = " "

# append reviews with label in docs list
# append reviews in strings for tokenize
for r in short_pos:
	docs.append( (r, "pos") )
	s_pos = s_pos+" "+r

for r in short_neg:
	docs.append( (r, "neg") )
	s_neg = s_neg+" "+r

for r in short_pos2.split('\n'):
	docs.append( (r, "pos") )
	s_pos = s_pos+" "+r

for r in short_neg2.split('\n'):
	docs.append( (r, "neg") )
	s_neg = s_neg+" "+r

# pickle docs for reuse and save time
docs_f = open("pickled_algo/documents.pickle", "wb")
pickle.dump(docs, docs_f)
docs_f.close()

# create empty list for store all words
all_words = []

# tokenize positive and negative reviews string and store in two different list
short_pos_words = word_tokenize(s_pos)
short_neg_words = word_tokenize(s_neg)

# all words from these lists append in all_words list
for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())

# A frequency distribution records the number of times each words
all_words = nltk.FreqDist(all_words)

# create a list of first 6000 keys of all_words
word_feature = list(all_words.keys())[:6000]

# pickle word_feature
word_feature_f = open("pickled_algo/word_features.pickle", "wb")
pickle.dump(word_feature, word_feature_f)
word_feature_f.close()

# create function which return dictionary of features
def find_features(doc):
	words = word_tokenize(doc)
	features = {}
	for w in word_feature:
		features[w] = (w in words)

	return features

# create featureset list in which we have tuple of reviews and category
featureset = [(find_features(rev), category) for (rev, category) in docs]

# pickle featureset
featureset_f = open("pickled_algo/featureset.pickle", "wb")
pickle.dump(featureset, featureset_f)
featureset_f.close()

# random shuffle featureset
random.shuffle(featureset)

# create training and testing data
training_set = featureset[:12000]
testing_set = featureset[12000:]

# create naive bayes classifier and train using training set
classifier = nltk.NaiveBayesClassifier.train(training_set)

#pickle naive bayes classifier
classifier_f = open("pickled_algo/naivebayes.pickle", "wb")
pickle.dump(classifier, classifier_f)
classifier_f.close()

#print accuracy of naive bayes classifier
print("Naive Bayes Accuracy :",(nltk.classify.accuracy(classifier, testing_set))*100)

# print most 15 informative features
classifier.show_most_informative_features(15)

# create Multinomial naive bayes classifier and train using training set
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

#pickle Multinomial naive bayes classifier
saveMNB_classifier = open("pickled_algo/MNBclassifier_features.pickle", "wb")
pickle.dump(MNB_classifier, saveMNB_classifier)
saveMNB_classifier.close()

#print accuracy of Multinomial naive bayes classifier
print("MNB_classifier Accuracy :",(nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# create Bernoulli naive bayes classifier and train using training set
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

#pickle Bernoulli naive bayes classifier
saveBNB_classifier = open("pickled_algo/BNBclassifier_features.pickle", "wb")
pickle.dump(BNB_classifier, saveBNB_classifier)
saveBNB_classifier.close()

#print accuracy of Bernoulli naive bayes classifier
print("BernoulliNB_classifier Accuracy :",(nltk.classify.accuracy(BNB_classifier, testing_set))*100)

# create LogisticRegression classifier and train using training set
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

#pickle LogisticRegression classifier
saveLOG_classifier = open("pickled_algo/LogisticRegression_classifier_features.pickle", "wb")
pickle.dump(LogisticRegression_classifier, saveLOG_classifier)
saveLOG_classifier.close()

#print accuracy of LogisticRegression classifier
print("LogisticRegression_classifier Accuracy :",(nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# create SGD classifier and train using training set
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

#pickle SGD classifier
saveSGDC_classifier = open("pickled_algo/SGDclassifier_features.pickle", "wb")
pickle.dump(SGDClassifier_classifier, saveSGDC_classifier)
saveSGDC_classifier.close()

#print accuracy of SGD classifier
print("SGDClassifier_classifier Accuracy :",(nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# create LinearSVC classifier and train using training set
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

#pickle LinearSVC classifier
saveLSVC_classifier = open("pickled_algo/LSVCclassifier_features.pickle", "wb")
pickle.dump(LinearSVC_classifier, saveLSVC_classifier)
saveLSVC_classifier.close()

#print accuracy of LinearSVC classifier
print("LinearSVC_classifier Accuracy :",(nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# create NuSVC classifier and train using training set
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

#pickle NuSVC classifier
saveNuSVC_classifier = open("pickled_algo/NuSVCclassifier_features.pickle", "wb")
pickle.dump(NuSVC_classifier, saveNuSVC_classifier)
saveNuSVC_classifier.close()

#print accuracy of NuSVC classifier
print("NuSVC_classifier Accuracy :",(nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# create voted_classifier which is object of VoteClassifier Class
voted_classifier = VoteClassifier(MNB_classifier, BNB_classifier, LogisticRegression_classifier, LinearSVC_classifier, NuSVC_classifier)

# print accuracy of voted_classifier
print("Voted_classifier Accuracy :",(nltk.classify.accuracy(voted_classifier, testing_set))*100)
