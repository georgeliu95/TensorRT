import os


def version(version_str):
    return tuple([int(num) for num in version_str.split(".")])


def check_file_non_empty(path):
    assert os.stat(path).st_size > 0
