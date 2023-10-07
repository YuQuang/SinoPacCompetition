import torch
import math
import pandas as pd

def cast_to_float(value) -> float:
    try:
        return 0.0 if math.isnan(float(value)) else float(value)
    except Exception:
        return 0.0

class SinoDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_path: str):
        self.__feature_cols = [
            "縣市", "鄉鎮市區", "路名", "土地面積", "使用分區", "移轉層次", "總樓層數",
            "主要用途", "主要建材", "建物型態", "屋齡", "建物面積", "車位面積",
            "車位個數", "橫坐標", "縱坐標", "備註", "主建物面積", "陽台面積", "附屬建物面積"
        ]
        self.__label_cols = [
            "單價"
        ]
        self.__df = pd.read_csv(csv_path)

    def __getitem__(self, index):
        feature_row = self.__df[self.__feature_cols].iloc[index].to_list()
        lable_row = self.__df[self.__label_cols].iloc[index].to_list()
        float_feature_row = list(map(
            cast_to_float,
            feature_row
        ))
        return torch.Tensor(float_feature_row), torch.Tensor(lable_row)
    
    def __len__(self):
        return self.__df.__len__()

class SinoTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_path: str):
        self.__feature_cols = [
            "縣市", "鄉鎮市區", "路名", "土地面積", "使用分區", "移轉層次", "總樓層數",
            "主要用途", "主要建材", "建物型態", "屋齡", "建物面積", "車位面積",
            "車位個數", "橫坐標", "縱坐標", "備註", "主建物面積", "陽台面積", "附屬建物面積"
        ]
        self.__label_cols = [
            "單價"
        ]
        self.__df = pd.read_csv(csv_path)

    def __getitem__(self, index):
        feature_row = self.__df[self.__feature_cols].iloc[index].to_list()
        float_feature_row = list(map(
            cast_to_float,
            feature_row
        ))
        return torch.Tensor(float_feature_row), 0
    
    def __len__(self):
        return self.__df.__len__()