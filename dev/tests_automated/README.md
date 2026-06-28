# Automatic Testing of Zero2Neuro Internals

This module provides some basic tests to ensure that our example or
test models function appropriately.  Each test only checks to make
sure that the data can be loaded and that training can start.  We
cannot easily check whether an appropriate level of performance is achieved.

## Process
1. ```cd tests_automated```
2. Activate python environment
3. ```pytest testall.py```

Pytest scans the local directory for all .sh files and executes each in sorted order.  Pytest should report that all tests have passed; if not, then we must investigate the reasons and fix either the implementation or the offending test.

