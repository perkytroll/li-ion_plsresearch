import numpy
import pandas as pd
from data_preprocessing import DataPreProcessing
from data_visualizer import DataVisualizer


class PCA(DataPreProcessing, DataVisualizer):
    def __init__(self):
        super().__init__()
        self.error_matrix = None
        self.score_space = None
        self.reconstructed_data = None
        self.eig_vectors = None
        self.eig_values = None

    def data_in_score_space(self, n_data: pd.DataFrame, num_components: int, num_of_comps: bool):
        cov_matrix = (n_data.T @ n_data) / (n_data.shape[1] - 1)
        self.eig_values, self.eig_vectors = numpy.linalg.eig(cov_matrix)
        idx = self.eig_values.argsort()[::-1]
        self.eig_vectors = self.eig_vectors[:, idx]

        if not num_of_comps:
            d_comps = self.eig_vectors[:, num_components - 1]
        else:
            d_comps = self.eig_vectors[:, 0: num_components]

        self.score_space = numpy.array(n_data @ d_comps)

        if numpy.ndim(self.score_space) == 1:
            self.reconstructed_data = self.score_space.reshape(-1, 1) @ d_comps.reshape(-1, 1).T
        else:
            self.reconstructed_data = self.score_space @ d_comps.T
        self.error_matrix = n_data - self.reconstructed_data

    def summary_statistics(self, n_data):
        error_squares = numpy.square(self.error_matrix)
        error_squares_sum = numpy.sum(numpy.array(error_squares))
        errors_variance = error_squares_sum / (error_squares.shape[0] * error_squares.shape[1] - 1)

        orig_data_squares = numpy.square(n_data)
        orig_data_squares_sum = numpy.sum(numpy.array(orig_data_squares))
        orig_data_variance = orig_data_squares_sum / (n_data.shape[0] * n_data.shape[1] - 1)

        r2 = 1 - (errors_variance / orig_data_variance)
        print("Value of R2 is \n", r2 * 100)

    def summary_statistics_column(self, n_data):
        r2_vals = []
        error_vals = []
        orig_data_vals = []
        for col in self.error_matrix:
            error_squares = numpy.square(self.error_matrix[col])
            error_squares_sum = numpy.sum(numpy.array(error_squares))
            errors_variance = error_squares_sum / (error_squares.shape[0] - 1)
            error_vals.append(errors_variance)

        for col_orig in n_data:
            orig_data_squares = numpy.square(n_data[col_orig])
            orig_data_squares_sum = numpy.sum(numpy.array(orig_data_squares))
            orig_data_variance = orig_data_squares_sum / (n_data.shape[0] - 1)
            orig_data_vals.append(orig_data_variance)

        if len(orig_data_vals) == len(error_vals):
            for ev, ov in zip(error_vals, orig_data_vals):
                r2_vals.append((1 - (ev / ov)) * 100)
        return r2_vals

    def plot_score_space(self):
        if self.score_space.shape[1] == 2:
            self.twod_plot_versus(self.score_space[:, 0], self.score_space[:, 1], 'T1 - T2 PLot', 'T1', 'T2')
        else:
            pass
