# gfan

Recognize machine-translated lyrics on [Netease Cloud Music](http://music.163.com/) using machine learning

## Usage

Fill in `data/source.txt` (actually a CSV file). Example formats are given. `is_correct=1` means this translation is
correct, and `is_correct=0` means it is highly suspect.

Then run:

```bash
python src/app.py
```

The default source language is Japanese (Baidu Translate: `jp`; Google Translate: `ja`), and the default
destination language is Chinese (Baidu Translate: `zh`; Google Translate: `zh-CN`). This can be configured
by calling `baidu_translate()` and `google_translate()` with different arguments.

*I damn hate Python's "mystic" import rules!*

## Requirements

Required Python version: **3.6+** (for variable annotations)

Packages:

- Net:
    - [requests](https://pypi.python.org/pypi/requests/)
    - [googletrans](https://pypi.python.org/pypi/googletrans/)
    - [hyper](https://pypi.python.org/pypi/hyper/) (optional)
- Machine learning general:
  - [numpy](https://pypi.python.org/pypi/numpy/)
  - [scikit-learn](https://pypi.python.org/pypi/scikit-learn/)
- Vectorization:
  - [jieba](https://pypi.python.org/pypi/jieba/)
  - [many-stop-words](https://pypi.python.org/pypi/many-stop-words/)

## Behind The Scenes

This program automatically tries to download the lyrics of given songs, and translates the lyrics using two major
Japanese-Chinese translating websites (Baidu Translate and Google Translate, esp. Baidu) used by Chinese users, to
generate training sets. Of course the dishonored human-made translations are categorized as `neg`, the poor ones.

## Q&A

### The name is strange!

gfan (pronunciation: *gee-fan*) reads similar to "机翻" (abbreviation of "机器翻译" in Chinese, lit. machine translation).

It is a mocking of "manual" machine translation.

### What is the motivation?

On Cloud Music poor translations can often be seen. A optimistic assumption is that, these low quality translations are
mainly contributed by people who does not actually understand the source language. Based on this assumption, those people
would use machine translations for help, and modify some words to make it more "natural".

Let's see how this works. This behavior is mostly seen on Japanese-Chinese translations. There are kanjis (jp: 漢字) in
Japanese, which are often the in the characters' traditional writing (zh: 繁体字) or variants (zh: 异体字) in modern Chinese.
So those people read the text, throw away characters they can't understand and reorganize the rest. An example is shown below.

**Lyrics (Japanese):** [source](http://music.163.com/#/song?id=28482417)

> …
>
> いつの日か　もう一度
>
> あなたが立ち上がれるやうに
>
> 世界を染め上げる
>
> 小さな花に成らう

**Translation (Chinese):**

> ……
>
> 总有一天 你会再次
>
> 如同被激励了一般
>
> 成为将世界染上色彩
>
> 的小小花朵吧

**Translation (English):**

> ...
>
> and one day in the future, you will once again
>
> become the small flowers
>
> that color the world
>
> as if you are inspired.

Please note that although the number of lines are all the same, order of clauses are adjusted
according to each language's customs. There are also ancient writings in lyrics (e.g. やうに/成らう→ように/成ろう in
modern Japanese; confirmed via web searching) that make it harder to translate.

Now let's see how the others perform.

**Baidu Translate:**

> 总有一天 (one day in the future)
>
> 宛如你振作 (as if you are cheered up)
>
> 染上了世界 (be painted with the world)
>
> 成了小小的花 (become small flowers)

**Google Translate:**

> 再次有一天 (again in one day)
>
> 所以，你站起来油 (so, you stand up oil)
>
> Someageru世界 (someageru world)
>
> 它将成为一个小的花 (it will become a small flower)

**Incorrect translation:** (on the same page of the lyrics)

> 总有一天 再一次 (oneday in the future, again)
>
> 宛若你屹立于此 (as if you stand still here)
>
> 染上了世界 (be painted with the world)
>
> 成了小小的花 (become small flowers)

You can obviously see the similarity between the incorrect translation and the text Baidu Translate prints out.

The differences can be easily explained too. That user knows a little Japanese so (s)he hears/sees "もう一度" (lit.
once again) ignored by Baidu Translate. This phrase frequently appears in animes so it is easy to recognize. "宛如你振作"
is not a correct sentence in Chinese, so (s)he reads "立ち上がれる" (lit. be cheered up, original form "立ち上がる" means
cheer up/stand up). But (s)he only knows two characters "立" (lit. stand) and "上" (lit. up/upon/above) that Chinese have,
and (s)he thinks "振作" (lit. cheer up) suggested by Baidu Translate is nonsense. After a little "recreation", (s)he
completes this clause as what you see ("宛若你屹立于此"), replacing "宛如" with "宛若" (their meanings are the same) and
extends "立" to "屹立" ("stand"→"stand still and firmly") to make it a more literary expression.

So translating by machine translation has a low cost and it is easy to operate: simply copy and paste, work done.
Although there are lyrics/translation reviewers on Cloud Music, their capabilities vary. With a low chance, machine
translated text does mislead reviewers, due to their intense work or carelessness. Some of these texts are manually
reorganized (like the example above) so people having no knowledge of that foreign language are easily deceived.

I write this little program to learn and recognize this kind of poor translation, as well as apply what I've learned during
the Data Mining course. However, [more complicated situations](https://zhuanlan.zhihu.com/p/22973727) still need human
reviewing.

Still, the translations can be corrected by re-uploading, though this procedure takes long time and its result is uncertain.
Some translations were poor (mostly machine translated) before, but they are replaced by new ones so the original text
is lost in the history. For example, you can still read comments complaining translation performed by machine, like
[this one](http://music.163.com/#/song?id=718438) (current version of translation is correct).

### So... how can I help?

Please provide more live examples of machine translations that passed reviewing! With a more accurate training set, this
program can recognize poor translations more precisely, even if they are manually edited.

## License

MIT License
