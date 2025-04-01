import os
import sys
import traceback
import contextlib
import io

"""
class Suppressor(object):

    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise Exception()

    def write(self, message): pass
        # self.stdout.write(message)

        # self.stdout.write(message)
    # def write(self, x): pass
    def flush(self):
        pass
"""


class Suppressor(object):

    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self.stdout = sys.stdout
            sys.stdout = self
        else:
            pass

    def __exit__(self, type, value, traceback):
        if self.suppress:
            sys.stdout = self.stdout
            if type is not None:
                raise Exception()
        else:
            pass

    def write(self, x): pass

    def flush(self): pass


"""
def flush(self, type, value, traceback):
    if self.suppress:
        sys.stdout = self.stdout
        if type is not None:
            raise Exception()
    else:
        pass
"""


def salute(name):
    """Says hi to someone."""
    print('Hi, {}!'.format(name))

# create a text trap and redirect stdout
text_trap = io.StringIO()
sys.stdout = text_trap

# execute our now mute function
salute('Anne')

# now restore stdout function
sys.stdout = sys.__stdout__


if __name__ == "__main__":

    def pf(target=sys.stdout):
        print("ballalalala")

    pf()

    with open(os.devnull, 'w') as devnull:
        pf(target=devnull)

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            pf()


    def pf2():
        print("ausdubasudbausd")


    with Suppressor(suppress=False):
        pf()

    with Suppressor(suppress=False):
        pf2()

    with Suppressor(suppress=True):
        pf2()
        pf()
        print("asdasd")

#
