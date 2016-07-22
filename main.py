import cPickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.stem.porter import PorterStemmer
from string import punctuation

from libs.twokenize import tokenizeRawTweetText
from tweet_classifier import TweetClassifier


stemmer = PorterStemmer()
stopwords_1 = set(stopwords.words('english'))
stopwords_2 = set([word.rstrip() for word in open('files/stopwords.txt', 'r')])
stopwords = list(set(stopwords_1 | stopwords_2))

def text_preprocessor(text):
    text = text.rstrip()
    # remove punctuation
    text = ''.join([char for char in text if char not in punctuation])
    # Stemming
    words = [stemmer.stem(w) for w in text.split()]

    return ' '.join(words)


if __name__ == '__main__':

    # classifier = TweetClassifier('datasets/traffic.csv')
    # print classifier.tune_parameters()
    # exit()

    pipeline = Pipeline([
                            ('vectorizer',  CountVectorizer(
                                                preprocessor=text_preprocessor,
                                                stop_words=stopwords,
                                                tokenizer=tokenizeRawTweetText,
                                                ngram_range=(1, 4),
                                                max_features=200,
                                                min_df=1,
                                                )),
                            ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                            ('classifier',  RandomForestClassifier(n_jobs=-1, criterion='entropy'))
                        ])


    classifier = TweetClassifier('datasets/traffic.csv', pipeline=pipeline)
    classifier.cross_validate(n_folds=5)
    classifier.train()

    test = ['Hello I am Chicago person who had an accident at home', 'Traffic sucks, car crashed near ashton lane']
    print classifier.predict(test)

    print 'Saving Model.....'
    filename = classifier.save_model('models')

    print 'Loading model and trying again....'
    model = cPickle.load(open('models/'+filename))

    print model.predict(test)
