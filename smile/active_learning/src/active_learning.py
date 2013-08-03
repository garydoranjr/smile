"""Implements class for active learning."""
import numpy as np

from progress import ProgressMonitor

class ActiveLearner(object):

    def __init__(self, classifier, queries, verbose=True):
        self.classifier = classifier
        self.queries = queries
        self.verbose = verbose

    def learn(self, X_labeled, y_labeled, X_pool, y_pool, X_test):
        # Initial Predictions
        self.classifier.fit(X_labeled, y_labeled)
        predictions = [self.classifier.decision_function(X_test)]

        if self.verbose:
            progress = ProgressMonitor(total=self.queries, msg='Active Learning')
        for q in range(self.queries):
            if len(X_pool) <= 0:
                if self.verbose: print 'Warning: skipping query %d...' % q
                predictions.append(predictions[-1])
            else:
                next_labeled = self.select(X_pool)
                X_labeled.append(X_pool.pop(next_labeled))
                y_labeled.append(y_pool.pop(next_labeled))
                self.classifier.fit(X_labeled, y_labeled)
                predictions.append(self.classifier.decision_function(X_test))
            if self.verbose: progress.increment()

        return predictions

    def select(self, pool):
        """
        Selects the instance from the pool whose labeled will be queried.
        @return : the index of the instance whose labeled will be queried
        """
        pass

class SVMActiveLearner(ActiveLearner):

    def __init__(self, *args, **kwargs):
        self.selection_technique = kwargs.pop('selection_technique', 'nearest')
        super(SVMActiveLearner, self).__init__(*args, **kwargs)

    def select(self, pool):
        if self.selection_technique == 'nearest':
            distances = np.abs(self.classifier.decision_function(pool))
            return np.argmin(distances)
        else:
            raise Exception('Unsupported selection technique: "%s"'
                            % self.selection_technique)
