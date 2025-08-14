#!/usr/bin/python

import os
import sys
from os.path import dirname, abspath, exists, splitext  # , basename
from os.path import join as join_path


def flatten_files(here):
    """Move all files in subdirectories to here, then delete subdirectories.
       Conflicting files are renamed, with 1 appended to their name."""
    for root, dirs, files in os.walk(here, topdown=False):
        if root != here:
            for name in files:
                source = join_path(root, name)
                target = handle_duplicates(join_path(here, name))
                os.rename(source, target)

        for name in dirs:
            os.rmdir(join_path(root, name))


def handle_duplicates(target):
    base, ext = splitext(target)
    count = 0
    while exists(target):
        count += 1
        target = base + repr(count) + ext
    return target


if __name__ == '__main__':
    this_dir = abspath(dirname(sys.argv[0]))
    flatten_files(this_dir)
