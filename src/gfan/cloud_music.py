import json
import logging
import os
import re
from typing import Union, List, Dict, Any

import requests

# Extracted from https://gist.github.com/3rogue/23c7bc2f876070dc3e956441e3d55a01
LYRIC_URL_TPL = "http://music.163.com/api/song/lyric?lv=-1&tv=-1&id={}"


class CloudMusicLyricResponseObject:
    class UserInfo:
        id: int
        status: int
        demand: int
        userid: int
        nickname: str
        uptime: int

    class Lyrics:
        version: int
        lyric: str

    uncollected: bool
    sgc: bool
    sfy: bool
    qfy: bool
    transUser: Union[UserInfo, None]
    lyricUser: Union[UserInfo, None]
    lrc: Union[Lyrics, None]
    tlyric: Union[Lyrics, None]
    code: int

    # Extra
    has_lyrics: bool
    has_translation: bool
    has_lyrics_user: bool
    has_translation_user: bool
    lyrics: List[str]
    translation: List[str]


def _build_cmlrc_object(d: Dict[Any, Any]) -> CloudMusicLyricResponseObject:
    def build_userinfo_object(d: Dict[Any, Any]) -> CloudMusicLyricResponseObject.UserInfo:
        r = CloudMusicLyricResponseObject.UserInfo()
        r.id = d["id"]
        r.status = d["status"]
        r.demand = d["demand"]
        r.userid = d["userid"]
        r.nickname = d["nickname"]
        r.uptime = d["uptime"]
        return r

    def build_lyrics_object(d: Dict[Any, Any]) -> CloudMusicLyricResponseObject.Lyrics:
        r = CloudMusicLyricResponseObject.Lyrics()
        r.version = d["version"]
        r.lyric = d["lyric"]
        return r

    r = CloudMusicLyricResponseObject()
    r.uncollected = d.get("uncollected", False)
    r.sgc = d["sgc"]
    r.sfy = d["sfy"]
    r.qfy = d["qfy"]

    r.has_lyrics = ("lrc" in d) and ("lyric" in d["lrc"]) and (d["lrc"]["lyric"] is not None)
    r.has_translation = ("tlyric" in d) and ("lyric" in d["tlyric"]) and (d["tlyric"]["lyric"] is not None)
    r.has_lyrics_user = "lyricUser" in d
    r.has_translation_user = "transUser" in d

    if r.has_lyrics:
        r.lrc = build_lyrics_object(d["lrc"])
    if r.has_lyrics_user:
        r.lyricUser = build_userinfo_object(d["lyricUser"])
    if r.has_translation:
        r.tlyric = build_lyrics_object(d["tlyric"])
    if r.has_translation_user:
        r.transUser = build_userinfo_object(d["transUser"])

    r.code = d["code"]

    def substr(s: str, sep: str, trim: bool = True) -> str:
        index = s.find(sep)
        if index < 0:
            s2: str = s
        else:
            s2: str = s[index + 1:]
        return s2.strip() if trim else s2

    if hasattr(r, "lrc") and len(r.lrc.lyric) > 0:
        lines = r.lrc.lyric.split("\n")
        r.lyrics = list(filter(lambda x: len(x) > 0, (substr(x, "]") for x in lines)))
    else:
        r.lyrics = []
    if hasattr(r, "tlyric") and len(r.tlyric.lyric) > 0:
        lines = r.tlyric.lyric.split("\n")
        r.translation = list(filter(lambda x: len(x) > 0, (substr(x, "]") for x in lines)))
    else:
        r.translation = []

    return r


def download_lyric(song_id: int, use_cache: bool = True, cache_dir="data/lrc", session: requests.Session = None) -> \
        Union[CloudMusicLyricResponseObject, None]:
    if use_cache:
        try:
            os.makedirs(cache_dir, mode=0o642)
        except IOError:
            pass

    cache_file_name = os.path.join(cache_dir, "{}.txt".format(song_id))

    if use_cache and os.path.exists(cache_file_name):
        with open(cache_file_name, encoding="utf-8") as fd:
            json_text: str = fd.read()
    else:
        url: str = LYRIC_URL_TPL.format(song_id)
        try:
            if session is None:
                r = requests.get(url)
            else:
                r = session.get(url)
            json_text: str = r.text
        except (requests.ConnectionError, requests.HTTPError, ValueError) as ex:
            logging.error(ex)
            return None

    if use_cache and not os.path.exists(cache_file_name):
        with open(cache_file_name, mode="w", encoding="utf-8") as fd:
            fd.write(json_text)

    try:
        d: Dict[Any, Any] = json.loads(json_text)
        obj: CloudMusicLyricResponseObject = _build_cmlrc_object(d)
        return obj
    except ValueError as ex:
        logging.error(ex)
        return None


def combine_translation(translation: List[str]) -> str:
    tr = "\n".join(translation)
    return tr


_cloud_music_id_format = re.compile(r"song\?id=([\d]+)")


def match_song_id(song_id: Union[int, str]) -> int:
    try:
        song_id = int(song_id)
    except ValueError:
        ids = _cloud_music_id_format.findall(song_id)
        song_id = ids[0] if len(ids) == 1 else None

    if song_id is None:
        raise ValueError("Invalid song ID: {}".format(song_id))
    if isinstance(song_id, str):
        song_id = int(song_id)

    return song_id
