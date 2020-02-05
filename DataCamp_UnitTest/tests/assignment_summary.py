import pytest
import numpy as np
import sys
import os

sys.path.append("/Users/XavierTang/Documents/Data Science/Python/python_basics/DataCamp_UnitTest/src")
from data.preprocessing_helpers import row_to_list, convert_to_int, preprocess
from models.train import split_into_training_and_testing_sets
from features.as_numpy import get_data_as_numpy_array

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

# # we can also use tmpdir
# @pytest.fixture
# def raw_and_clean_data_file(tmpdir):
#     raw_data_file_path = tmpdir.join("raw.txt")
#     clean_data_file_path = tmpdir.join("clean.txt")
#     with open("raw.txt", "w") as f:
#         f.write("1,801\t201,411\n"
#                "1,767565,112\n"
#                "2,002\t333,209\n"
#                "1990\t7822,911\n"
#                "1,285\t389129\n")
#     yield raw_data_file_path, clean_data_file_path
#     #no teardown needed

# #test
# def test_on_raw_data(raw_and_clean_data_file):
#     raw_path, clean_path = raw_and_clean_data_file
#     preprocess(raw_path, clean_path)
#     with open(clean_data_file_path) as f:
#         lines = f.readlines()
#         first_line = lines[0]
#         assert first_line == "1801\t201411\n"
#         second_line = lines[1]
#         assert second_line == "2002\t333209\n"

# Add a decorator to make this function a fixture
@pytest.fixture
def clean_data_file():
    file_path = "clean_data_file.txt"
    with open(file_path, "w") as f:
        f.write("201\t305671\n7892\t298140\n501\t738293\n")
    yield file_path
    os.remove(file_path)
    
# Pass the correct argument so that the test can use the fixture
def test_on_clean_file(clean_data_file):
    expected = np.array([[201.0, 305671.0], [7892.0, 298140.0], [501.0, 738293.0]])
    # Pass the clean data file path yielded by the fixture as the first argument
    actual = get_data_as_numpy_array(clean_data_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual) 
