import numpy as np
from scipy.stats import norm
from typing import Union

class _NormlizerConvert:
    def convert_in_to_machine_data(self, x):
        raise NotImplementedError
    def recovery_from_machine_data(self, x):
        raise NotImplementedError
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Min_Max_Convert(_NormlizerConvert):
    def __init__(self, _min=None, _max=None):
        assert _min is not None and _max is not None, "min and max must be provided"
        self.min = _min
        self.max = _max
    def convert_in_to_machine_data(self, value):
        return (value - self.min) / (self.max - self.min)
    def recovery_from_machine_data(self, value):
        return value * (self.max - self.min) + self.min
    def __repr__(self):
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"

class Unit_Convert(_NormlizerConvert):
    def __init__(self, unit=None):
        assert unit is not None, "unit must be provided"
        self.unit = unit
    def convert_in_to_machine_data(self, value):
        return value/self.unit
    def recovery_from_machine_data(self, value):
        return value*self.unit
    def __repr__(self):
        return f"{self.__class__.__name__}(unit={self.unit})"
    
class Gaussian_Convert(_NormlizerConvert):
    def __init__(self, mean=None, std=None):
        assert mean is not None and std is not None, "mean and std must be provided"
        self.mean = mean
        self.std = std
    def convert_in_to_machine_data(self, value):
        return (value - self.mean) / self.std
    def recovery_from_machine_data(self, value):
        return value * self.std + self.mean
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})" 

class Normal_Distribution_Convert(_NormlizerConvert):
    def __init__(self, data:Union[str, np.ndarray]=None):
        assert data is not None, "data must be provided"
        if isinstance(data, str):
            data = np.load(data)
        self.sorted_data = np.sort(data)
        self.ecdf_values = np.arange(1, len(data) + 1) / (len(data) + 1)
    def convert_in_to_machine_data(self, value):
        rank = np.searchsorted(self.sorted_data, value, side='left') + 1
        empirical_cdf_value = rank / (len(self.sorted_data) + 1)
        normal_value = norm.ppf(empirical_cdf_value)
        return normal_value
    def recovery_from_machine_data(self, normal_value):
        percentile = norm.cdf(normal_value)
        index = np.searchsorted(self.ecdf_values, percentile, side='left') - 1
        original_value = self.sorted_data[index] # <--- it seem it always return the value that is smaller than the percentile
        return original_value
    

class ABSCoordinate_Convert(_NormlizerConvert):
    def __init__(self, unit):
        self.unit = unit
    def convert_in_to_machine_data(self, value):
        return np.abs(value)/self.unit
    def recovery_from_machine_data(self, normal_value):
        return normal_value*self.unit
def get_normlizer_convert(normlizer:str=None, **kwargs)->_NormlizerConvert:
    if normlizer is None:
        return Unit_Convert(unit=1)
    if normlizer == "min_max":
        return Min_Max_Convert(_min=kwargs["_min"], _max=kwargs["_max"])
    elif normlizer == "unit":
        return Unit_Convert(unit=kwargs["unit"])
    elif normlizer == "gaussian":
        return Gaussian_Convert(mean=kwargs["mean"], std=kwargs["std"])
    elif normlizer == "normal_distribution":
        return Normal_Distribution_Convert(data=kwargs["data"])
    elif normlizer == "abs":
        return ABSCoordinate_Convert(unit=1)
    elif normlizer == "absunit":
        return ABSCoordinate_Convert(unit=kwargs["unit"])

    else:
        raise ValueError(f"normlizer={normlizer} is not supported")


