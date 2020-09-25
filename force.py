import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool, cv
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.multiclass import OneVsRestClassifier
import glob
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.utils import class_weight
import collections
import os


def get_score(A, y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S / y_true.shape[0]

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true))-0.5)
    plt.ylim(len(np.unique(y_true))-0.5, -0.5)
    np.set_printoptions(precision=2)
    plt.savefig(title+'.png')
    return ax

def plot_feature_importance(clf):
    fea_imp = pd.DataFrame({'imp': clf.feature_importances_, 'col': clf.feature_names_})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
    plt.title('Feature Importance')
    plt.ylabel('Features')
    plt.xlabel('Importance')
    plt.savefig('feature_importance.png', bbox_inches='tight')



class Model(object):
    def _preprocess(self, features, train=True, add_seq_features=False, add_poly_features=False):
        wells = features['WELL'].unique()
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(features[['DEPTH_MD', 'Z_LOC']])
        features['Z_LOC'] = imp.transform(features[['DEPTH_MD', 'Z_LOC']])[:, 1]

        interpolate_features = ['X_LOC', 'Y_LOC', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF',
                                'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']

        for well in wells:
            features.loc[features.WELL == well, interpolate_features] = \
                features.loc[features.WELL == well, interpolate_features].interpolate().bfill('rows').ffill('rows')
        if train:
            features.FORCE_2020_LITHOFACIES_CONFIDENCE.fillna(1, inplace=True)

        features['BS'] = round(features.BS, 0)
        BS_list = [12, 8, 18, 15, 6, 10, 26]
        for i in range(len(features)):
            bs_value = features.BS[i]
            if ((~np.isnan(bs_value)) and (bs_value not in BS_list)):
                features.loc[i, 'BS'] = min(BS_list, key=lambda x: abs(x - bs_value))
        features_fe = features

        if add_seq_features:
            seq_column_names = ['RHOB', 'GR', 'NPHI', 'DTC', 'SP', 'ROP', 'DRHO']
            seq_length = 5
            features_fe = pd.DataFrame()
            for w in np.unique(wells):
                print(w)
                tmp = features[features['WELL'] == w]
                for i in range(1, seq_length + 1):
                    sf_df = tmp[seq_column_names].shift(i)
                    sf_df.columns = [x + '_sf_' + str(i) for x in seq_column_names]
                    tmp = tmp.join(sf_df).bfill('rows')
                for seq in seq_column_names:
                    tmp[seq + '_grad'] = np.gradient(tmp[seq].rolling(center=False, window=1).mean())
                features_fe = features_fe.append(tmp)

        features_fe = features_fe.astype({"GROUP": str, "FORMATION": str, "BS": str})
        features_fe.loc[:, ['GROUP', 'FORMATION', 'BS']] = features_fe[['GROUP', 'FORMATION', 'BS']].fillna('unknown')
        features_fe.fillna(features_fe.median(), inplace=True)

        if add_poly_features:
            poly_features = ['RHOB', 'GR', 'NPHI', 'DTC', 'SP',  'ROP', 'DRHO']
            deg = 2
            poly = PolynomialFeatures(deg, interaction_only=False, include_bias=False)
            tmp = poly.fit_transform(features_fe[poly_features])
            feature_names = poly.get_feature_names(poly_features)
            tmp = pd.DataFrame(tmp, columns=feature_names)
            features_fe = features_fe[features_fe.columns.difference(poly_features)].join(tmp)

        if train:
            features_fe.to_csv('data_fe_train.zip', index=False, compression='gzip')

        return features_fe

    def train(self, lithology_numbers, model_name, fe=False, add_seq_features=True, add_poly_features = False):
        data = pd.read_csv('train.csv', sep=';')
        data = data.drop(columns=['SGR'])

        if fe:
            X = data[list(data.columns[:-2]) + ['FORCE_2020_LITHOFACIES_CONFIDENCE']].copy()
            X = self._preprocess(X, train=True, add_seq_features=add_seq_features, add_poly_features=add_poly_features)
        else:
            X = pd.read_csv('data_fe_train.zip', compression='gzip')

        y = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].copy()

        print(X.shape, y.shape)
        y = y.map(lithology_numbers)
        class_names = y.unique()
        weigths = class_weight.compute_class_weight(class_weight='balanced', classes=class_names, y=y)
        X = X.drop(columns=['FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL'])

        category_to_lithology = {y: x for x, y in lithology_numbers.items()}


        if model_name == 'catboost.joblib.gz':
            cat_features = [s for s in X.columns if ('GROUP' in s) or ('BS' in s) or ('FORMATION' in s)]
            X[cat_features] = X[cat_features].astype(str)
            cat_features_index = [X.columns.get_loc(c) for c in cat_features if c in X]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            best = {'eval_metric': 'TotalF1', 'depth': 10, 'l2_leaf_reg': 5, 'learning_rate': 0.15,
                    'loss_function': 'MultiClass', 'od_type': 'Iter', 'early_stopping_rounds': 100,
                    'task_type': 'GPU', 'verbose': 500, 'random_seed': 42, 'cat_features': cat_features_index}
            clf = OneVsRestClassifier(CatBoostClassifier(**best), n_jobs =2).fit(X_train, y_train)
        elif model_name == 'xgboost.joblib.gz':
            onehot_X = pd.get_dummies(X[['GROUP', 'FORMATION', 'BS']])
            X = X.drop(columns=['GROUP', 'FORMATION', 'BS']).join(onehot_X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(X_train)
            # joblib.dump(scaler, 'scaler.joblib.gz', compress=('gzip', 3))
            # X_train = scaler.transform(X_train)
            # X_test = scaler.transform(X_test)

            clf = OneVsRestClassifier(XGBClassifier(tree_method='gpu_hist', learning_rate=0.12,
                                      max_depth=3,
                                      min_child_weight=10,
                                      n_estimators=150,
                                      seed=43,
                                      colsample_bytree=0.9), n_jobs =2).fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        A = np.load('penalty_matrix.npy')
        score = get_score(A, y_test.values, y_pred)
        print("\n score: {}".format(score))
        print(classification_report(y_test, y_pred))
        plot_confusion_matrix(y_true=np.vectorize(category_to_lithology.get)(y_test),
                                  y_pred=np.vectorize(category_to_lithology.get)(y_pred),
                                  classes=list(category_to_lithology.values()))
        joblib.dump(clf, model_name, compress=('gzip', 3))
        # plot_feature_importance(clf)

    def predict(self, model_name, features, add_seq_features=True, add_poly_features=False):
        X = self._preprocess(features, train=False, add_seq_features=add_seq_features, add_poly_features = add_poly_features)
        X = X.drop(columns=['WELL'])
        if model_name == 'xgboost.joblib.gz':
            onehot_X = pd.get_dummies(X[['GROUP', 'FORMATION', 'BS']])
            X = X.drop(columns=['GROUP', 'FORMATION', 'BS']).join(onehot_X)
        return joblib.load(model_name).predict(X)

if __name__ == '__main__':
    train = True
    test = True
    model = 'catboost'

    feature_engineering = True
    add_seq_features = False
    add_poly_features = False

    model_name = model +'.joblib.gz'
    lithology_numbers = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11}

    model = Model()
    if train:
        model.train(lithology_numbers, model_name, fe=feature_engineering, add_seq_features=add_seq_features, add_poly_features=add_poly_features)
    if test:
        open_test_features = pd.read_csv('test.csv', sep=';')
        open_test_features = open_test_features.drop(columns=['SGR'])
        test_prediction = model.predict(model_name, open_test_features, add_seq_features=add_seq_features, add_poly_features=add_poly_features)
        category_to_lithology = {y: x for x, y in lithology_numbers.items()}
        test_prediction_for_submission = np.vectorize(category_to_lithology.get)(test_prediction)
        print(test_prediction_for_submission)
        np.savetxt('test_predictions.csv', test_prediction_for_submission, header='lithology', fmt='%i', comments='')