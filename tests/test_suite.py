# Test functionality of chainlink classes and functions

from __future__ import print_function
import os, sys
import numpy as np
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

class TestNormalizationMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNormalizationMethods, self).__init__(*args, **kwargs)

        self.test_chunk1 = np.asarray([[50, 50, 20, 10, 40, 60, 70], 
                                       [60, 60, 20, 10, 40, 60, 70]], 
                                      dtype=np.int32).T

        self.test_chunk2 = np.asarray([[50, 52, 20, 13, 42, 60, 100], 
                                       [60, 77, 20, 27, 40, 61, 100]], 
                                      dtype=np.int32).T
                                      

    def test_max_normalization(self):
        answer = np.asarray([[35, 36, 14, 9,  29, 42, 70], 
                             [42, 53, 14, 18, 28, 42, 70]], 
                            dtype=np.int32).T

        result = cl.normalize_chunk(self.test_chunk1, self.test_chunk2, 'Max')

        # Assert that the normalized array appears as expected
        np.testing.assert_array_equal(result, answer)

        # Assert that the datatype of the new array matches the old array
        self.assertEqual(self.test_chunk2.dtype, result.dtype)
        

    def test_stdev_normalization(self):
        pass


if __name__ == '__main__':
    unittest.main()