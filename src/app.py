try:
    # Make PyCharm's intellisense work.
    from .gfan import preprocess, train_sklearn, train_tf, eval_tf
except ImportError:
    # This is what Python really recognizes.
    from gfan import preprocess, train_sklearn, train_tf, eval_tf

TEST_SKLEARN = False
TEST_TF = True

if __name__ == '__main__':
    if TEST_SKLEARN or TEST_TF:
        preprocess()

    if TEST_SKLEARN:
        train_sklearn()

    if TEST_TF:
        # Extra evaluation dataset
        preprocess("data/eval", "data/source_eval.txt")

        # ts = train_tf()
        ts = 1497961163
        eval_tf(ts)
