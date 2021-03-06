from typing import Set
from many_stop_words import get_stop_words


def read_all_stop_words() -> Set[str]:
    # Data source: https://wenku.baidu.com/view/7ca26338376baf1ffc4fad6a.html
    with open("data/chinese_stop_words.txt", mode="r", encoding="utf-8") as local_file:
        text_lines = local_file.readlines()
        text_lines = list(x.replace("\n", "") for x in text_lines)

    with open("data/chinese_stop_symbols.txt", mode="r", encoding="utf-8") as local_file:
        symbol_lines = local_file.readlines()
        symbol_lines = list(x.replace("\n", "") for x in symbol_lines)

    public_stop_words = get_stop_words("zh")

    stop_words: Set[str] = set()
    stop_words = stop_words.union(text_lines)
    stop_words = stop_words.union(symbol_lines)
    stop_words = stop_words.union(public_stop_words)

    return stop_words


def read_all_stop_symbols() -> Set[str]:
    with open("data/chinese_stop_symbols.txt", mode="r", encoding="utf-8") as local_file:
        symbol_lines = local_file.readlines()
        symbol_lines = list(x.replace("\n", "") for x in symbol_lines)

    stop_words: Set[str] = set(symbol_lines)

    return stop_words
