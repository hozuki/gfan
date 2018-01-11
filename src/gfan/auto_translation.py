import json
from typing import List, Dict

import requests
from googletrans import Translator

from .cloud_music import CloudMusicLyricResponseObject

BAIDU_FANYI_URL = "http://fanyi.baidu.com/v2transapi"


class BaiduTranslationResult:
    class SimplifiedDataEntry:
        dst: str
        src: str

    def __init__(self, root: Dict[str, any]):
        self.data = []

        if root["error"] is not None and int(root["error"]):
            # Baidu now has two extra fields, "token" and "sign".
            # "token" is statically written (as inline <script> in HTML) into 'window.common' object.
            # "sign" hashes some request fields.
            # I don't want to waste my time analyzing this *, because I don't have the need to do it.
            return

        o = root["trans_result"]
        self.lang_from = o["from"]
        self.lang_to = o["to"]
        self.status = int(o["status"])
        for entry in o["data"]:
            e2 = BaiduTranslationResult.SimplifiedDataEntry()
            e2.dst = entry["dst"]
            e2.src = entry["src"]
            self.data.append(e2)

    lang_from: str
    lang_to: str
    status: int
    data: List[SimplifiedDataEntry]


def baidu_translate(o: CloudMusicLyricResponseObject, lang_from: str = "jp", lang_to: str = "zh",
                    session: requests.Session = None) -> List[str]:
    if not o.has_lyrics:
        return []
    lyrics_text = "\n".join(o.lyrics)

    data_dict: Dict[str, str] = {
        "from": lang_from,
        "to": lang_to,
        "query": lyrics_text,
        "trans_type": "realtime",
        "simple_means_flag": str(3)
    }

    # if session is not None:
    #     session.headers["User-Agent"] = \
    #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)" \
    #         " Chrome/59.0.3071.104 Safari/537.36"
    #     session.headers["Referer"] = "http://fanyi.baidu.com"
    #     session.headers["Origin"] = "http://fanyi.baidu.com"
    #     session.headers["X-Requested-With"] = "XMLHttpRequest"

    try:
        if session is None:
            r = requests.post(BAIDU_FANYI_URL, data=data_dict)
        else:
            r = session.post(BAIDU_FANYI_URL, data=data_dict)
    except (requests.HTTPError, requests.HTTPError):
        return []

    json_text = r.text
    try:
        obj: Dict[str, any] = json.loads(json_text, encoding="utf-8")
    except ValueError:
        raise

    translation_obj = BaiduTranslationResult(obj)
    result = [x.dst for x in translation_obj.data]
    return result


def google_translate(o: CloudMusicLyricResponseObject, lang_from: str = "ja", lang_to: str = "zh-CN",
                     session: requests.Session = None) -> List[str]:
    if not o.has_lyrics:
        return []
    lyrics_text = "\n".join(o.lyrics)

    # urls = ("http://translate.google.com", "http://translate.google.cn", "http://translate.google.nl")
    urls = None
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.104 Safari/537.36"
    client = Translator(service_urls=urls, user_agent=ua)
    translated = client.translate(lyrics_text, dest=lang_to, src=lang_from)
    translated_lines = translated.text.split("\\n")

    return translated_lines
