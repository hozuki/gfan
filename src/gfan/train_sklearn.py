import numpy as np
from sklearn.datasets import load_files as load_sklearn_data_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from . import preprocess, read_all_stop_words


def train_sklearn() -> None:
    stop_word_set = read_all_stop_words()
    lrc_training = load_sklearn_data_files("data/training")

    # Please use "neg" and "pos" for category names, keeping them the same as directory names in `data/training`.
    labels = [("neg", "Suspect machine translation"), ("pos", "Natural translation")]
    labels.sort()
    label_names = [x[1] for x in labels]

    for _ in range(10):
        print("------")

        terms_train, terms_test, y_train, y_test = train_test_split(lrc_training.data, lrc_training.target,
                                                                    test_size=0.2)

        count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words=stop_word_set)
        x_train: np.ndarray = count_vec.fit_transform(terms_train)
        x_test: np.ndarray = count_vec.transform(terms_test)

        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        print("True positive rate: {}".format(np.mean(y_pred == y_test)))

        test_set_size = len(y_test)
        print("Test set size: ", test_set_size)

        print("* Before probability correction")
        bins = np.bincount(y_pred)
        bins_len = len(bins)
        for i, label_name in enumerate(label_names):
            if i < bins_len:
                print("Count of class {} ({}): {}".format(i, label_name, bins[i]))

        # precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
        class_prob = clf.predict_proba(x_test)
        class_prob = class_prob[:, 0]  # based on the assumption that categories are [neg, pos]

        report = np.ndarray((test_set_size,), dtype=np.int)
        for i in range(test_set_size):
            prob: np.float64 = class_prob[i]
            # Threshold 0.8 (all predictions are class 0) is... counter-intuitive.
            # The model should have worked better.
            # The right way is using clf.classes_ and read class_prob[i] to decide the class index,
            # but since our probabilities are a lot higher than normal (0.5), we have to perform a custom filtering.
            if prob > 0.8:
                cls = 0  # neg
            else:
                cls = 1  # pos
            report[i] = cls

        print("* After probability correction")
        bins = np.bincount(report)
        bins_len = len(bins)
        for i, label_name in enumerate(label_names):
            if i < bins_len:
                print("Count of class {} ({}): {}".format(i, label_name, bins[i]))

        print("* Actual")
        bins = np.bincount(y_test)
        bins_len = len(bins)
        for i, label_name in enumerate(label_names):
            if i < bins_len:
                print("Count of class {} ({}): {}".format(i, label_name, bins[i]))

        print()
        print(classification_report(y_test, report, target_names=label_names))


if __name__ == '__main__':
    preprocess()
    train_sklearn()
