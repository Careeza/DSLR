import pytest
import numpy as np
from tools import ft_count, ft_sum, ft_mean, ft_std, ft_max, ft_min, ft_percentile

class TestBasicStats:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.simple_list = [1, 2, 3, 4, 5]
        self.with_nan = [1, np.nan, 3, np.nan, 5]
        self.all_nan = [np.nan, np.nan, np.nan]
        self.empty_list = []
        self.float_precision = 1e-6

    def test_count(self):
        assert ft_count(self.simple_list) == 5
        assert ft_count(self.with_nan) == 3
        assert ft_count(self.all_nan) == 0
        assert ft_count(self.empty_list) == 0

    def test_sum(self):
        assert abs(ft_sum(self.simple_list) - 15) < self.float_precision
        assert abs(ft_sum(self.with_nan) - 9) < self.float_precision
        assert abs(ft_sum(self.all_nan) - 0) < self.float_precision
        assert abs(ft_sum(self.empty_list) - 0) < self.float_precision

    def test_mean(self):
        assert abs(ft_mean(self.simple_list) - 3) < self.float_precision
        assert abs(ft_mean(self.with_nan) - 3) < self.float_precision
        assert np.isnan(ft_mean(self.all_nan))
        assert np.isnan(ft_mean(self.empty_list))

class TestDeviation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.simple_list = [1, 2, 3, 4, 5]
        self.with_nan = [1, np.nan, 3, np.nan, 5]
        self.all_nan = [np.nan, np.nan, np.nan]
        self.empty_list = []
        self.float_precision = 1e-6

    def test_std_population(self):
        assert abs(ft_std(self.simple_list) - np.std(self.simple_list)) < self.float_precision
        assert abs(ft_std(self.with_nan) - np.std([1, 3, 5])) < self.float_precision
        assert np.isnan(ft_std(self.all_nan))
        assert np.isnan(ft_std(self.empty_list))

    def test_std_sample(self):
        assert abs(ft_std(self.simple_list, ddof=1) - np.std(self.simple_list, ddof=1)) < self.float_precision
        assert abs(ft_std(self.with_nan, ddof=1) - np.std([1, 3, 5], ddof=1)) < self.float_precision

class TestExtremes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.simple_list = [1, 2, 3, 4, 5]
        self.with_nan = [1, np.nan, 3, np.nan, 5]
        self.all_nan = [np.nan, np.nan, np.nan]
        self.empty_list = []
        self.float_precision = 1e-6

    def test_max(self):
        assert abs(ft_max(self.simple_list) - 5) < self.float_precision
        assert abs(ft_max(self.with_nan) - 5) < self.float_precision
        assert ft_max(self.all_nan) is None
        assert ft_max(self.empty_list) is None

    def test_min(self):
        assert abs(ft_min(self.simple_list) - 1) < self.float_precision
        assert abs(ft_min(self.with_nan) - 1) < self.float_precision
        assert ft_min(self.all_nan) is None
        assert ft_min(self.empty_list) is None

class TestPercentiles:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.simple_list = [1, 2, 3, 4, 5]
        self.with_nan = [1, np.nan, 3, np.nan, 5]
        self.all_nan = [np.nan, np.nan, np.nan]
        self.empty_list = []
        self.float_precision = 1e-6

    def test_quartiles(self):
        assert abs(ft_percentile(self.simple_list, 0) - 1) < self.float_precision
        assert abs(ft_percentile(self.simple_list, 0.25) - 2) < self.float_precision
        assert abs(ft_percentile(self.simple_list, 0.5) - 3) < self.float_precision
        assert abs(ft_percentile(self.simple_list, 0.75) - 4) < self.float_precision
        assert abs(ft_percentile(self.simple_list, 1) - 5) < self.float_precision

    def test_with_nan(self):
        assert abs(ft_percentile(self.with_nan, 0.5) - 3) < self.float_precision

    def test_invalid_cases(self):
        assert np.isnan(ft_percentile(self.all_nan, 0.5))
        assert np.isnan(ft_percentile(self.empty_list, 0.5))
        assert np.isnan(ft_percentile(self.simple_list, -0.1))
        assert np.isnan(ft_percentile(self.simple_list, 1.1))

class TestFloatingPointPrecision:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.float_precision = 1e-6
        self.float_numbers = [1.1, 2.2, 3.3, 4.4, 5.5]
        self.periodic_float = [1/3, 2/3, 1/6, 1/9]

    def test_sum_precision(self):
        expected_sum = sum(self.float_numbers)
        assert abs(ft_sum(self.float_numbers) - expected_sum) < self.float_precision

        expected_periodic_sum = sum(self.periodic_float)
        assert abs(ft_sum(self.periodic_float) - expected_periodic_sum) < self.float_precision

    def test_mean_precision(self):
        expected_mean = sum(self.float_numbers) / len(self.float_numbers)
        assert abs(ft_mean(self.float_numbers) - expected_mean) < self.float_precision

        expected_periodic_mean = sum(self.periodic_float) / len(self.periodic_float)
        assert abs(ft_mean(self.periodic_float) - expected_periodic_mean) < self.float_precision

    def test_std_precision(self):
        expected_std = np.std(self.float_numbers)
        assert abs(ft_std(self.float_numbers) - expected_std) < self.float_precision

        expected_periodic_std = np.std(self.periodic_float)
        assert abs(ft_std(self.periodic_float) - expected_periodic_std) < self.float_precision

class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.float_precision = 1e-6

    def test_single_value(self):
        single_value = [42.42]
        assert abs(ft_mean(single_value) - 42.42) < self.float_precision
        assert abs(ft_std(single_value)) < self.float_precision
        assert abs(ft_max(single_value) - 42.42) < self.float_precision
        assert abs(ft_min(single_value) - 42.42) < self.float_precision
        assert abs(ft_percentile(single_value, 0.5) - 42.42) < self.float_precision

    def test_extreme_numbers(self):
        large_numbers = [1e15, 2e15, 3e15]
        assert abs(ft_mean(large_numbers) - 2e15) < 1e9
        assert abs(ft_max(large_numbers) - 3e15) < 1e9
        assert abs(ft_min(large_numbers) - 1e15) < 1e9

        small_numbers = [1e-15, 2e-15, 3e-15]
        assert abs(ft_mean(small_numbers) - 2e-15) < 1e-21
        assert abs(ft_max(small_numbers) - 3e-15) < 1e-21
        assert abs(ft_min(small_numbers) - 1e-15) < 1e-21