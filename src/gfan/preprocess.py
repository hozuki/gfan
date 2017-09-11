import csv
import os
import time
from typing import List, Tuple

import jieba
import requests

from . import baidu_translate, google_translate, match_song_id, download_lyric, combine_translation


# Use MeCab for Japanese tokenization. (@sangsi)

def preprocess(base_dir: str = "data/training", source_file: str = "data/source.txt",
               skip_existing: bool = True) -> None:
    lyrics_session, baidu_session, google_session = requests.Session(), requests.Session(), None

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

        base_path = os.path.join(base_dir, category)
        os.makedirs(base_dir, mode=0o642, exist_ok=True)

        # Save manual translation
        fn = "{}.txt".format(song_id)
        fn = os.path.join(base_path, fn)
        if not os.path.exists(fn) or not skip_existing:
            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)

        # Prepare to save results from auto translation tools.
        """From here on, all training sets are 'negative'."""
        auto_translated = False
        base_path = os.path.join(base_dir, "neg")
        os.makedirs(base_path, mode=0o642, exist_ok=True)

        # Save Baidu Translate results
        fn = "{}_baidu.txt".format(song_id)
        fn = os.path.join(base_path, fn)
        print("> Translating by Baidu Translate")
        if not os.path.exists(fn) or not skip_existing:
            baidu_translated = baidu_translate(lyric_obj, session=baidu_session)
            translation = combine_translation(baidu_translated)
            segs = jieba.cut(translation, cut_all=False)
            translation = " ".join(segs)

            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)

            auto_translated = True
        else:
            print("(Skipped)")

        # Save Google Translate results
        fn = "{}_google.txt".format(song_id)
        fn = os.path.join(base_path, fn)
        print("> Translating by Google Translate")
        if not os.path.exists(fn) or not skip_existing:
            google_translated = google_translate(lyric_obj, session=google_session)
            translation = combine_translation(google_translated)
            segs = jieba.cut(translation, cut_all=False)
            translation = " ".join(segs)

            with open(fn, mode="w", encoding="utf-8") as fd:
                fd.write(translation)

            auto_translated = True
        else:
            print("(Skipped)")

        if auto_translated:
            time.sleep(0.5)

    print("Preprocessing complete.")
