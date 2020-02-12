# Test functionality of chainlink classes and functions

from __future__ import print_function
import os, sys
import unittest

# Get directory location of the test_suite script
test_suite_dir = os.path.dirname(os.path.realpath(__file__))

# Chainlink should be a directory above the test_suite dir
chainlink_dir = str(os.sep).join(test_suite_dir.split(os.sep)[:-1])

# Add chainlink directory to system path
sys.path.append(chainlink_dir)

# Import Chainlink
import chainlink as cl 


# TODO: Write tests