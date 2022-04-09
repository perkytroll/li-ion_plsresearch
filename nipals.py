import random
import numpy
import pandas as pd
from data_preprocessing import DataPreProcessing
from data_visualizer import DataVisualizer


class NIPALS(DataPreProcessing, DataVisualizer):
    def __init__(self):
        super().__init__()
        self.normalized_data_x: pd.DataFrame = None
        self.normalized_data_y: pd.DataFrame = None

    def set_x_y_blocks(self):
        self.normalized_data_x = self.normalized_x_block
        self.normalized_data_y = self.normalized_y_block

    @staticmethod
    def regressor(x, y):
        xy_sum = numpy.sum(x * y)
        sq = x.transpose() @ x
        coeff = xy_sum / sq
        return coeff[0][0]

    @staticmethod
    def unit_mag_loadings(loading_frame: pd.DataFrame):
        temp_loading_norm = numpy.sqrt(numpy.square(loading_frame).sum(axis=0))
        temp_loading = loading_frame / temp_loading_norm
        return temp_loading

    def nipals_iteration_implementation(self):
        updater: bool = True
        temp_score_x = pd.DataFrame(columns=['t1'])
        temp_loadings_x = pd.DataFrame(columns=['w1'])
        independent_score_x: pd.DataFrame = pd.DataFrame(columns=['t1'])
        independent_loadings_x = pd.DataFrame(columns=['w1'])

        temp_score_y = pd.DataFrame(columns=['u1'])
        temp_loadings_y = pd.DataFrame(columns=['c1'])
        independent_score_y: pd.DataFrame = self.normalized_data_y.iloc[:, random.randint(0, 2)]
        independent_loadings_y = pd.DataFrame(columns=['c1'])

        while updater:
            for col in self.normalized_data_x:
                temp_loadings_x.loc[col] = self.regressor(independent_score_y.values.reshape(-1, 1),
                                                          self.normalized_data_x[col].values.reshape(-1, 1))

            normalized_loading = self.unit_mag_loadings(temp_loadings_x)
            independent_loadings_x = normalized_loading
            temp_loadings_x = temp_loadings_x.iloc[0:0]

            for index, row in self.normalized_data_x.iterrows():
                temp_score_x.loc[index] = ((row.values.reshape(1, -1) @ independent_loadings_x).iloc[0, 0]) / \
                                          ((independent_loadings_x.transpose() @ independent_loadings_x).iloc[0, 0])

            independent_score_x = temp_score_x
            temp_score_x = pd.DataFrame(columns=['t1'])

            for col in self.normalized_data_y:
                temp_loadings_y.loc[col] = self.regressor(independent_score_x.values.reshape(-1, 1),
                                                          self.normalized_data_y[col].values.reshape(-1, 1))

            independent_loadings_y = temp_loadings_y
            temp_loadings_y = temp_loadings_y.iloc[0:0]

            for index, row in self.normalized_data_y.iterrows():
                temp_score_y.loc[index] = ((row.values.reshape(1, -1) @ independent_loadings_y).iloc[0, 0]) / \
                                          ((independent_loadings_y.transpose() @ independent_loadings_y).iloc[0, 0])

            independent_score_y = temp_score_y
            temp_score_y = pd.DataFrame(columns=['u1'])
            print("OK")

