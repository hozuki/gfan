try:
    # Make PyCharm's intellisense work.
    from .gfan import train_dataset, preprocess
except ImportError:
    # This is what Python really recognizes.
    from gfan import train_dataset, preprocess

if __name__ == '__main__':
    preprocess()
    train_dataset()
