import unittest

from src import download_lyric, google_translate


class TestGoogleTranslation(unittest.TestCase):
    def test_translation(self):
        song_id = 441116283
        lo = download_lyric(song_id, use_cache=False)
        translation = google_translate(lo)
        self.assertTrue(len(translation) > 0)


if __name__ == '__main__':
    unittest.main()
