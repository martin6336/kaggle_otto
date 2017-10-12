"""
5-fold cv - log loss 0.447489661199
"""
import graphlab as gl
import numpy as np
import logging
import os

from hyperopt import fmin, hp, tpe

from sklearn.base import BaseEstimator
from sklearn import ensemble

from otto_utils import consts, utils


MODEL_NAME = 'model_14_bagging_xgboost'
MODE = 'cv'  # cv|submission|holdout|tune

logging.disable(logging.INFO)


class XGBoost(BaseEstimator):
    def __init__(self, max_iterations=50, max_depth=9, min_child_weight=4, row_subsample=.75,
                 min_loss_reduction=1., column_subsample=.8, step_size=.3, verbose=True):
        self.n_classes_ = None
        self.classes_ = None
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.row_subsample = row_subsample
        self.min_loss_reduction = min_loss_reduction
        self.column_subsample = column_subsample
        self.step_size = step_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        sf = self._array_to_sframe(X, y)
        self.model = gl.boosted_trees_classifier.create(sf, target='target',
                                                        max_iterations=self.max_iterations,
                                                        max_depth=self.max_depth,
                                                        min_child_weight=self.min_child_weight,
                                                        row_subsample=self.row_subsample,
                                                        min_loss_reduction=self.min_loss_reduction,
                                                        column_subsample=self.column_subsample,
                                                        step_size=self.step_size,
                                                        validation_set=None,
                                                        verbose=self.verbose)

        return self

    def predict(self, X):
        preds = self.predict_proba(X)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        sf = self._array_to_sframe(X)
        preds = self.model.predict_topk(sf, output_type='probability', k=self.n_classes_)

        return self._preds_to_array(preds)

    # Private methods
    def _array_to_sframe(self, data, targets=None):
        d = dict()
        for i in xrange(data.shape[1]):
            d['feat_%d' % (i + 1)] = gl.SArray(data[:, i])
        if targets is not None:
            d['target'] = gl.SArray(targets)

        return gl.SFrame(d)

    def _preds_to_array(self, preds):
        p = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
        p['id'] = p['id'].astype(int) + 1
        p = p.sort('id')
        del p['id']
        preds_array = np.array(p.to_dataframe(), dtype=float)

        return preds_array


if __name__ == '__main__':
    train, labels, test, _, _ = utils.load_data()

    clf_xgb = XGBoost(max_iterations=300, max_depth=12, min_child_weight=4.9208250938262745, row_subsample=.9134478530382129,
                      min_loss_reduction=.5132278416508804, column_subsample=.730128689911957, step_size=.1)
    clf = ensemble.BaggingClassifier(base_estimator=clf_xgb, n_estimators=20,
                                     max_samples=1., max_features=1., bootstrap=True,
                                     n_jobs=1, verbose=True, random_state=23)
    # max_sample,max_feature对应bootstrap,bootstrap_feature.如果为真就是不放回的，这里的意思就是不重复的抽取所有样本，放回的抽取feature数目的特征。

    if MODE == 'cv':
        scores, predictions = utils.make_blender_cv(clf, train, labels, calibrate=False)
        print 'CV:', scores, 'Mean log loss:', np.mean(scores)
        utils.write_blender_data(consts.BLEND_PATH, MODEL_NAME + '.csv', predictions)
    elif MODE == 'submission':
        clf.fit(train, labels)
        predictions = clf.predict_proba(test)
        utils.save_submission(consts.DATA_SAMPLE_SUBMISSION_PATH,
                              os.path.join(consts.ENSEMBLE_PATH, MODEL_NAME + '.csv'),
                              predictions)
    elif MODE == 'holdout':
        score = utils.hold_out_evaluation(clf, train, labels, calibrate=False)
        print 'Log loss:', score
    elif MODE == 'tune':
        # Objective function
        def objective(args):
            max_depth, min_child_weight, row_subsample, min_loss_reduction, column_subsample = args
            clf = XGBoost(max_depth=max_depth, min_child_weight=min_child_weight,
                          row_subsample=row_subsample, min_loss_reduction=min_loss_reduction,
                          column_subsample=column_subsample, verbose=False)
            score = utils.hold_out_evaluation(clf, train, labels, calibrate=False)
            print 'max_depth, min_child_weight, row_subsample, min_loss_reduction, column_subsample, logloss'
            print args, score
            return score
        # Searching space
        space = (
            hp.quniform('max_depth', 12, 12, 1),
            hp.uniform('min_child_weight', .5, 10.),
            hp.uniform('row_subsample', .3, 1.),
            hp.uniform('min_loss_reduction', .1, 3.),
            hp.uniform('column_subsample', .1, 1.),
        )

        best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=500)
        print 'Best solution:', best_sln
    else:
        print 'Unknown mode'
