import re
import warnings
import pandas as pd
import numpy as np
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from itertools import combinations
from sklearn import decomposition, pipeline, metrics


def clean_data(data):
    '''
    Stems and removes stop words from training and test data
    '''
    stemmer = SnowballStemmer('english')
    stop = stopwords.words('english')
    for column_name in ['query', 'product_title', 'product_description']:
        for index, row in data.iterrows():
            warnings.filterwarnings('error')
            try:
                extracted_data = (' ').join(
                    [i for i in BeautifulSoup(row[column_name], 'lxml')
                    .get_text(' ')
                    .split(' ')
                    ])
            except UserWarning:
                pass
            cleaned_data = re.sub('[^a-zA-Z0-9]',' ', extracted_data)
            stemmed_data = (' ').join(
                [stemmer.stem(i) for i in cleaned_data.split(' ')
                ])
            remove_stop_words = ('').join(
                [i for i in stemmed_data if i not in stop]
                )
            data.set_value(index, column_name, unicode(remove_stop_words))
    return data

def feature_extract(train, test):
    '''
    Feature engineering to add more relevance to query, product title and product description
    '''
    X_train = train['query'] + ' ' + train['product_title'] + ' ' + train['product_description']
    y_train = train['median_relevance']
    X_test = test['query'] + ' ' + test['product_title'] + ' ' + test['product_description']
    if 'median_relevance' in test.columns.values:
        y_test = test['median_relevance']
    else:
        y_test = test['id']
    return X_train, y_train, X_test, y_test

def tuner(vectorizer, param_grid, dataset, model):
    X_train, y_train, _, _ = feature_extract(dataset, dataset)
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, 
                                        greater_is_better = True)
    
    tuner_model = GridSearchCV(estimator=model, 
                                param_grid=param_grid, 
                                scoring=kappa_scorer,
                                verbose=10, 
                                n_jobs=-1, 
                                iid=True, 
                                refit=True, 
                                cv=5)

    vectorizer.fit(X_train)
    X_train =  vectorizer.transform(X_train)
    tuner_model.fit(X_train, y_train)
    print("Best score: %0.3f" % tuner_model.best_score_)
    print("Best parameters set:")
    best_parameters = tuner_model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def perform_cross_validation(vectorizer, model, train, test):
    '''
    Performs a kfold cross validation of a given model
    '''
    kfold_train_test = []
    extracted_features = []
    kf = StratifiedKFold(train_clean["query"], n_folds=5)
    for train_index, test_index in kf:
        train_kfold = train_clean.loc[train_index]
        test_kfold = train_clean.loc[test_index]
        extracted_features.append(tuple(feature_extract(train_kfold, test_kfold)))

    score_count = 0
    score_total = 0.0
    submission = []
    print model
    for X_train, y_train, X_test, y_test in extracted_features:
        vectorizer.fit(X_train)
        X_train =  vectorizer.transform(X_train) 
        X_test = vectorizer.transform(X_test)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_data = pd.DataFrame({'id': y_test, 'predictions': predictions})
        submission.append(test_data)
        score_count += 1
        score = quadratic_weighted_kappa(y = y_test, y_pred = predictions)
        score_total += score
        print("Kfold score " + str(score_count) + ": " + str(score))
    average_score = score_total/float(score_count)
    print("Average score: " + str(average_score))
    return submission

def perform_predictions(vectorizer, model, train, test):
    '''
    Performs the final prediction on test dataset
    '''
    submission = []
    X_train, y_train, X_test, y_test = feature_extract(train, test)
    vectorizer.fit(X_train)
    train_features = vectorizer.transform(X_train)
    test_features = vectorizer.transform(X_test)
    model.fit(train_features, y_train)
    final_predictions = model.predict(test_features)
    test_data = pd.DataFrame({'id': y_test, 'predictions': final_predictions})
    submission.append(test_data)
    return submission

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    '''
    Returns the confusion matrix between rater's ratings
    '''
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    '''
    Returns the counts of each type of rating that a rater made
    '''
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    '''
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    '''
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

if __name__ == '__main__':
    print 'Loading data...'
    train_raw = pd.read_csv('data/train.csv').fillna('')
    test_raw = pd.read_csv('data/test.csv').fillna('')

    print 'Cleaning data...'
    train_clean = clean_data(train_raw)
    test_clean = clean_data(test_raw)

    # pickle.dump(train_clean, open('data/train_clean.pkl', 'w'))
    # pickle.dump(test_clean, open('data/test_clean.pkl', 'w'))

    # train_clean = pickle.load(open('data/train_clean.pkl', 'r'))
    # test_clean = pickle.load(open('data/test_clean.pkl', 'r'))

    vectorizer = TfidfVectorizer(
        min_df=1,
        max_features=None,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 4),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1
        )

    print 'Training...'
    pipeline_1 = Pipeline([
        ('svd', TruncatedSVD(n_components=400)),
        ('scl', StandardScaler()),
        ('svm', SVC(C=10))
        ])
    model_1 = MultinomialNB(alpha=0.0015)
    model_2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
    
    # print 'Tuning...'
    # param_grid = {
    #     'alpha': np.arange(0.0, 0.05, 0.0001)
    #     }
    # tuner(vectorizer, param_grid, train_clean, model_1)

    cv_pred_1 = perform_cross_validation(vectorizer, pipeline_1, train_clean, test_clean)
    cv_pred_2 = perform_cross_validation(vectorizer, model_1, train_clean, test_clean)
    cv_pred_3 = perform_cross_validation(vectorizer, model_2, train_clean, test_clean)
    
    print 'Predicting...'
    pred_1 = perform_predictions(vectorizer, pipeline_1, train_clean, test_clean)
    pred_2 = perform_predictions(vectorizer, model_1, train_clean, test_clean)
    pred_3 = perform_predictions(vectorizer, model_2, train_clean, test_clean)

    # pickle.dump(cv_pred_1, open('data/cv_pred_1.pkl', 'w'))
    # pickle.dump(cv_pred_2, open('data/cv_pred_2.pkl', 'w'))
    # pickle.dump(cv_pred_3, open('data/cv_pred_3.pkl', 'w'))
    # pickle.dump(pred_1, open('data/pred_1.pkl', 'w'))
    # pickle.dump(pred_2, open('data/pred_2.pkl', 'w'))
    # pickle.dump(pred_3, open('data/pred_3.pkl', 'w'))

    # cv_pred_1 = pickle.load(open('data/cv_pred_1.pkl', 'r'))
    # cv_pred_2 = pickle.load(open('data/cv_pred_2.pkl', 'r'))
    # cv_pred_3 = pickle.load(open('data/cv_pred_3.pkl', 'r'))
    # pred_1 = pickle.load(open('data/pred_1.pkl', 'r'))
    # pred_2 = pickle.load(open('data/pred_2.pkl', 'r'))
    # pred_3 = pickle.load(open('data/pred_3.pkl', 'r'))

    print 'Ensembling...'
    cv_preds = [cv_pred_1, cv_pred_2, cv_pred_3]
    wt_list = combinations(np.arange(0,1.05,0.05),3)
    wt_final = []
    for w in wt_list:
        if sum(w) == 1.0:
            wt_final.append(w)
    max_average_score = 0
    max_weights = None
    for wt in wt_final:
        total_score = 0
        for i in range(3):
            y_true = cv_preds[0][i]['id']
            weighted_prediction = sum([wt[x] * cv_preds[x][i]['predictions'].astype(int).reset_index() for x in range(3)])
            weighted_prediction = [round(p) for p in weighted_prediction['predictions']]
            total_score += quadratic_weighted_kappa(y_true, weighted_prediction)
        average_score = total_score/3.0
        if average_score > max_average_score:
            max_average_score = average_score
            max_weights = wt
    print 'Best set of weights: ' + str(max_weights)
    print 'Corresponding score: ' + str(max_average_score)

    preds = [pred_1, pred_2, pred_3]
    weighted_prediction = sum([max_weights[x] * preds[x][0]['predictions'].astype(int) for x in range(3)])
    weighted_prediction = [int(round(p)) for p in weighted_prediction]
    submission = pd.DataFrame({'id': test_clean['id'], 'prediction': weighted_prediction})
    submission.to_csv('submission.csv', index=False)
