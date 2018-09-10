import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def from_root(path):
    return os.path.join(ROOT, path)
