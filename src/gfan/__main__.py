import csv
import os
import sys
import time
from typing import List, Tuple

import jieba
import numpy as np
import requests
from sklearn.datasets import load_files as load_sklearn_data_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from . import baidu_translate, google_translate, match_song_id, download_lyric, combine_translation, read_all_stop_words


# Use MeCab for Japanese tokenization. (@sangsi)

def preprocess(skip_existing: bool = True) -> None:
    base_dir_tpl = "data/training/{}"
    lyrics_session, baidu_session, google_session = requests.Session(), requests.Session(), None

    source_file = "data/source.txt"
    # is_correct, id
    case_list: List[Tuple[bool, int]] = list()
    with open(source_file, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(reader):
            if i > 0:  # Filter out the header line.
                is_correct = bool(int(row[0]))
                song_id = match_song_id(row[1])
                case_list.append((is_correct, song_id))

    total_cast_count = len(case_list)
    for i, case in enumerate(case_list):
        song_id = case[1]
        print("Preprocessing {}/{} (id={})".format(i + 1, total_cast_count, song_id))

        print("> Downloading lyrics")
        lyric_obj = download_lyric(song_id, session=lyrics_session)
        translation = combine_translation(lyric_obj.translation)

        print("> Vectorizing")
        segs = jieba.cut(translation, cut_all=False)
        translation = " ".join(segs)

        category = "pos" if case[0] else "neg"

        base_dir = base_dir_tpl.format(category)
        os.makedirs(base_dir, mode=0o642, exist_ok=True)

        # Save manual translation
        fn = "{}.txt".format(song_id)
        fn = os.path.join(base_dir, fn)
        if not os.path.exists(fn) or not skip_existing:
            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)

        # Prepare to save results from auto translation tools.
        """From here on, all training sets are 'negative'."""
        base_dir = base_dir_tpl.format("neg")
        os.makedirs(base_dir, mode=0o642, exist_ok=True)

        # Save Baidu Translate results
        fn = "{}_baidu.txt".format(song_id)
        fn = os.path.join(base_dir, fn)
        print("> Translating by Baidu Translate")
        if not os.path.exists(fn) or not skip_existing:
            baidu_translated = baidu_translate(lyric_obj, session=baidu_session)
            translation = combine_translation(baidu_translated)
            segs = jieba.cut(translation, cut_all=False)
            translation = " ".join(segs)

            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)
        else:
            print("(Skipped)")

        # Save Google Translate results
        fn = "{}_google.txt".format(song_id)
        fn = os.path.join(base_dir, fn)
        print("> Translating by Google Translate")
        if not os.path.exists(fn) or not skip_existing:
            google_translated = google_translate(lyric_obj, session=google_session)
            translation = combine_translation(google_translated)
            segs = jieba.cut(translation, cut_all=False)
            translation = " ".join(segs)

            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)
        else:
            print("(Skipped)")

        time.sleep(0.5)

    print("Preprocessing complete.")


def train_dataset() -> None:
    stop_word_set = read_all_stop_words()
    lrc_training = load_sklearn_data_files("data/training")

    for _ in range(10):
        print("------")

        terms_train, terms_test, y_train, y_test = train_test_split(lrc_training.data, lrc_training.target,
                                                                    test_size=0.34)

        count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words=stop_word_set)
        x_train = count_vec.fit_transform(terms_train)
        x_test = count_vec.transform(terms_test)

        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)

        print("True positive rate: {}".format(np.mean(predicted == y_test)))

        # precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
        true_probabilities = clf.predict_proba(x_test)[:, 1]
        report = true_probabilities > 0.5
        print(classification_report(y_test, report, target_names=["neg", "pos"]))


if __name__ == '__main__':
    preprocess()
    train_dataset()
