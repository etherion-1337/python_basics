import pytest
import numpy as np
import sys

sys.path.append("/Users/XavierTang/Documents/Data Science/Python/python_basics/DataCamp_UnitTest/src")
from data.preprocessing_helpers import row_to_list, convert_to_int
from models.train import split_into_training_and_testing_sets

def test_for_clean_row():
	assert row_to_list("2,081\t314,942\n") == ["2,081", "314,942"]

def test_for_missing_area():
	assert row_to_list("\t293,410\n") is None

def test_for_missing_tab():
	assert row_to_list("1,463238,765\n") is None

def test_on_string_with_one_comma():
  assert convert_to_int("2,081") == 2081

def test_for_missing_area_with_message():
	actual = row_to_list("\t293,410\n")
	expected = None
	message = ("row_to_list('\t293,410\n') returned {0} instead of {1}". format(actual, expected))
	assert actual is expected, message

# multiple assert in one unit test
def test_on_string_with_one_comma_multiple():
	return_value = convert_to_int("2,081")
	assert isinstance(return_value, int) # assert the return is an int at all
	assert return_value == 2081

# test on raising error correctly
# if func split_into_training_and_testing_sets raises expected ValueError, it will be silenced and test will pass
# if func is buggy and does not raise ValueError, test will fail
def test_valueerror_on_one_dimensional_argument():
	example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])

	with pytest.raises(ValueError):
		split_into_training_and_testing_sets(example_argument)

def test_valueerror_on_one_dimensional_argument_message():
	example_argument = np.array([2081, 314942, 1059, 186606, 1148, 206186])

	# exception_info stores ValueError
	with pytest.raises(ValueError) as exception_info:
		split_into_training_and_testing_sets(example_argument)

	assert exception_info.match("Argument data_array must be two dimensional. Got 1 dimensional array instead!")


class TestRowToList(object): #start with Test, then use CamelCase
    def test_on_no_tab_no_missing_value(self):    # (0, 0) boundary value
    # Assign actual to the return value for the argument "123\n"
        actual = row_to_list("123\n")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
    def test_on_two_tabs_no_missing_value(self):    # (2, 0) boundary value
        actual = row_to_list("123\t4,567\t89\n")
    # Complete the assert statement
        assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
    def test_on_one_tab_with_missing_value(self):    # (1, 1) boundary value
        actual = row_to_list("\t4,567\n")
    # Format the failure message
        assert actual is None, "Expected: None, Actual: {0}".format(actual)


# Declare the test class
class TestSplitIntoTrainingAndTestingSets(object):
    # Fill in with the correct mandatory argument
    def test_on_one_row(self):
        test_argument = np.array([[1382.0, 390167.0]])
        with pytest.raises(ValueError) as exc_info:
            split_into_training_and_testing_sets(test_argument)
        expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
