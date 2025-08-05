import numpy as np


class ZScoreNormalization:
    @staticmethod
    def normalization(feature: np.ndarray, mean, std):
        std[(0.0000001 >= std) & (std >= -0.0000001)] = 1  # change
        feature = (feature - mean) / std
        # for i in range(feature.shape[1]):
        #     # print(i)
        #     if std[i] == 0:  # NOTE: if std is zero, zscore doesn't make sense. #origin:if mean[i] == 0 and std[i] == 0:
        #         feature[:, i] = 0
        return feature

    @staticmethod
    def inverse_normalization(feature: np.ndarray, mean, std):
        feature = feature * std + mean
        return feature
