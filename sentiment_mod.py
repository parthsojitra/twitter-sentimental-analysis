# import all needed packages 
import nltk
from nltk.tokenize import word_tokenize
import pickle
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
		

# load pickled docs
docs_f = open("pickled_algo/documents.pickle", "rb")
docs = pickle.load(docs_f)
docs_f.close()

# load pickled word_feature
word_feature_f = open("pickled_algo/word_features.pickle", "rb")
word_feature = pickle.load(word_feature_f)
word_feature_f.close()

# create function which return dictionary of features
def find_features(doc):
	words = word_tokenize(doc)
	features = {}
	for w in word_feature:
		features[w] = (w in words)

	return features


# load pickled featureset
featureset_f = open("pickled_algo/featureset.pickle", "rb")
featureset = pickle.load(featureset_f)
featureset_f.close()

# load pickled naive bayes classifier
classifier_f = open("pickled_algo/naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


# load pickled Multinomial naive bayes classifier
saveMNB_classifier = open("pickled_algo/MNBclassifier_features.pickle", "rb")
MNB_classifier = pickle.load(saveMNB_classifier)
saveMNB_classifier.close()


# load pickled Bernoulli naive bayes classifier
saveBNB_classifier = open("pickled_algo/BNBclassifier_features.pickle", "rb")
BNB_classifier = pickle.load(saveBNB_classifier)
saveBNB_classifier.close()


# load pickled LogisticRegression classifier
saveLOG_classifier = open("pickled_algo/LogisticRegression_classifier_features.pickle", "rb")
LogisticRegression_classifier = pickle.load(saveLOG_classifier)
saveLOG_classifier.close()


# load pickled SGD classifier
saveSGDC_classifier = open("pickled_algo/SGDclassifier_features.pickle", "rb")
SGDClassifier_classifier = pickle.load(saveSGDC_classifier)
saveSGDC_classifier.close()


# load pickled LinearSVC classifier
saveLSVC_classifier = open("pickled_algo/LSVCclassifier_features.pickle", "rb")
LinearSVC_classifier = pickle.load(saveLSVC_classifier)
saveLSVC_classifier.close()


# load pickled NuSVC classifier
saveNuSVC_classifier = open("pickled_algo/NuSVCclassifier_features.pickle", "rb")
NuSVC_classifier = pickle.load(saveNuSVC_classifier)
saveNuSVC_classifier.close()


# cerate voted_classifier which is object of VoteClassifier Class
voted_classifier = VoteClassifier(MNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)

# create function for find sentiment of some text
# function return maximum occured vote and confidence
def sentiment(text):
	feats = find_features(text)

	return voted_classifier.classify(feats), voted_classifier.confidence(feats)
