import unittest
from typing import List

from src import download_lyric, CloudMusicLyricResponseObject


class TestLyricsDownload(unittest.TestCase):
    def test_downloads(self):
        test_song_ids = [
            30413521,  # Pure music
            618845,  # Not pure music, no lyrics, no translation
            631141,  # Not pure music, lyrics, no translation
            480097172,  # Not pure music, lyrics and translation
        ]
        results: List[CloudMusicLyricResponseObject] = []
        for song_id in test_song_ids:
            result = download_lyric(song_id, use_cache=False)
            results.append(result)
        self.assertTrue(len(results[0].lyrics) == 0 and len(results[0].translation) == 0)
        self.assertTrue(len(results[1].lyrics) == 0 and len(results[1].translation) == 0)
        self.assertTrue(len(results[2].lyrics) > 0 and len(results[2].translation) == 0)
        self.assertTrue(len(results[3].lyrics) > 0 and len(results[3].translation) > 0)


if __name__ == '__main__':
    unittest.main()
