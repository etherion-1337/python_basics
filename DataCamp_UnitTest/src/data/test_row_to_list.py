import pytest

from data.preprocessing_helpers import row_to_list, convert_to_int

def test_for_clean_row():
	assert row_to_list("2,081\t314,942\n") == ["2,081", "314,942"]

def test_for_missing_area():
	assert row_to_list("\t293,410\n") is None

def test_for_missing_tab():
	assert row_to_list("1,463238,765\n") is None

def test_on_string_with_one_comma():
  assert convert_to_int("2,081") == 2081
