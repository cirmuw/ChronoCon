""" example_module_illustrting_pytest.py

Test driven development can be a livesaver for your project. 
It also makes other people who want to reuse your code more comftable. 

There are seveal frameworks for testing. 
Here we will illustrate how to use `pytest` is also integrated in out CI-pipline (see `.gitlab-ci.yml`)
see https://doc.pytest.org/en/latest/index.html for the documentation 

*Note:*
As an alternative you might also consider using `unittest`, especially if you have a large complicated project.
https://docs.python.org/3/library/unittest.html#module-unittest


**Note:** 
Running  `pytest` will only pick up files wich follow the standard testing convention (e.g. matching `test_*.py` or `*_test.py`)
One way to go is to have two files. E.g. `transforms.py` and `transforms_test.py`.
For simpliciy we keep it there as one file. 

If you prefere fewer files you can still run the tests by specifing the filename explicitly  (e.g. `pytest  transforms.py`).

To run all the test in the modul use: 
```bash
  pytest --doctest-modules  --verbose
```
This will also run the doctest, otherwise only the pytest parts will we run.
"""

import pytest


def add_1(x):
    return x + 1

# Testing can simply be done with asserts
def test_add_1():
    assert add_1(3) == 4
    
    
# Check that a certain exception is raised
def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()
        
        
# add a doctest instead: 
def add_2(x):
    """Add 2 -- to illustrate a doctest

    Examples
    --------
    >>> add_2(0)
    2
    """
    return x + 2


# You can also skip parts of the examples
def add_3(x):
    """Add 3 -- to illustrate a doctest

    Examples
    --------
    >>> random.random()  # doctest: +SKIP
    0.156231223
    
    >>> add_3(0)
    3
    """
    return x + 3




# Or let's test a class: 
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
 
    def get_area(self):
        area = self.width * self.height
        if area < 0: 
            return -1 # use -1 as 'error code'; Not pretty, but common
        else: 
            return area
 
    def set_width(self, width):
        self.width = width
 
    def set_height(self, height):
        self.height = height
 
# The test function to be executed by PyTest
def test_Rectangle_normal_case():
    rectangle = Rectangle(2, 3)
    assert rectangle.get_area() == 6, "incorrect area"
    
    
    
# It might also make sense to group tests together
class TestGetAreaRectangle:
    def test_normal_case(self):
        rectangle = Rectangle(2, 3)
        assert rectangle.get_area() == 6, "incorrect area"
    def test_negative_case(self): 
        """expect -1 as output to denote error when looking at negative area"""
        rectangle = Rectangle(-1, 2)
        assert rectangle.get_area() == -1, "incorrect negative output"