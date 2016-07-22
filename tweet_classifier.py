import re
import json
import time
import cPickle
import numpy as np
import pandas as pd
from ttp import ttp
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.learning_curve import learning_curve

from twokenize import tokenizeRawTweetText


stemmer = PorterStemmer()
stopwords_1 = set(stopwords.words('english'))
stopwords_2 = set([word.rstrip() for word in open('stopwords.txt', 'r')])
stopwords = list(set(stopwords_1 | stopwords_2))

def text_preprocessor(text):
    text = text.rstrip()
    # remove punctuation
    text = ''.join([char for char in text if char not in punctuation])
    # Stemming
    words = [stemmer.stem(w) for w in text.split()]

    return ' '.join(words)


class TweetClassifier(object):

    def __init__(self, data_path, sep=',', label_field='label', text_field='text', \
                 model=None, pipeline=None, test_params=None):

        if model is None:
            self.model = RandomForestClassifier(n_jobs=-1)
        else:
            self.model = model

        self.sep = sep
        self.data_path = data_path
        self.label_field = label_field
        self.text_field = text_field
        self.data = self.load_data()

        if test_params is None:
            self.test_params = {
                            'vectorizer__preprocessor': (text_preprocessor, None),
                            'vectorizer__ngram_range': ((1, 2), (1, 3), (1, 4)),
                            'vectorizer__min_df': (1, 5, 10, 20),
                            'vectorizer__max_features': (50, 100, 200),
                            'tfidf__use_idf': (True, False),
                            'tfidf__norm': ('l1', 'l2', None),
                            'classifier__criterion': ('gini', 'entropy'),
                          }
        else:
            self.test_params = test_params

        if pipeline is None:
            self.pipeline = Pipeline([
                                        ('vectorizer',  CountVectorizer(
                                                            stop_words=stopwords,
                                                            tokenizer=tokenizeRawTweetText,
                                                            )),
                                        ('tfidf', TfidfTransformer() ),
                                        ('classifier',  self.model)
                                    ])
        else:
            self.pipeline = pipeline


    def shuffle_dataset(self):
        return self.data.reindex(np.random.permutation(self.data.index))

    def split_data(self, percentage=0.8, shuffle=True):
        msk = np.random.rand(len(self.data)) < percentage
        if shuffle:
            self.data = self.shuffle_dataset()

        return self.data[msk], self.data[~msk]

    def load_data(self):
        return pd.read_csv(self.data_path, sep=self.sep)[[self.text_field, self.label_field]]

    def train(self):
        train_text = self.data[self.text_field].values
        train_y = self.data[self.label_field].values
        self.pipeline.fit(train_text, train_y)

    def tune_parameters(self, params=None):
        if params is None:
            params = self.test_params

        msg_train, msg_test, label_train, label_test = \
            train_test_split(self.data[self.text_field].values, self.data[self.label_field].values, test_size=0.2)

        grid = GridSearchCV(
                    self.pipeline,  # pipeline from above
                    params,  # parameters to tune via cross validation
                    refit=True,  # fit using all available data at the end, on the best found param combination
                    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
                    scoring='accuracy',  # what score are we optimizing?
                    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
                )
        tuner = grid.fit(msg_train, label_train)
        # with open('tuning.json', 'w') as out:
        #     out.write(tuner.grid_scores_+'\n')

        maximun = 0
        index = 0
        for i, r in enumerate(tuner.grid_scores_):
            if r[1] > maximun:
                index = i
                maximun = r[1]

        return tuner.grid_scores_[index]

    def cross_validate(self, n_folds=5, pos_label=1):
        k_fold = KFold(n=len(self.data), n_folds=n_folds)
        scores = []
        confusion = np.array([[0, 0], [0, 0]])

        for train_indices, test_indices in k_fold:
            train_text = self.data.iloc[train_indices][self.text_field].values
            train_y = self.data.iloc[train_indices][self.label_field].values
            test_text = self.data.iloc[test_indices][self.text_field].values
            test_y = self.data.iloc[test_indices][self.label_field].values

            # self.train(train_text, train_y)
            # predictions = self.predict(test_text)
            self.pipeline.fit(train_text, train_y)
            predictions = self.pipeline.predict(test_text)

            confusion += confusion_matrix(test_y, predictions)
            score = f1_score(test_y, predictions, pos_label=pos_label)
            scores.append(score)

        print('Total instances classified:', len(self.data))
        print('Score:', sum(scores)/len(scores))
        print('Confusion matrix:')
        print(confusion)

    def save_model(self, _folder):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = 'model_'+timestr+'.pkl'
        with open(_folder+'/'+filename, 'wb') as fout:
            cPickle.dump(self.pipeline, fout)
        return filename

    def load_model(self, path):
        self.pipeline = cPickle.load(open(path))

    def predict(self, instances):
        return self.pipeline.predict(instances)
