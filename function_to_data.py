import pickle
import traceback
import warnings

from scipy.optimize import curve_fit
import pandas as pd
import numpy
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution


class FunctionToData:
    def __init__(self):
        self.constants = None
        self.master_frame = pd.DataFrame(columns=['Element Name', 'Atomic Mass', 'Partial Charge', 'R2', 'A', 'B', 'C'])
        self.init_constants = [0.2, 1, 0]
        self.master_file_uri: str = 'raw_data.xlsx'
        with open('element_pc_dict.pkl', 'rb') as f:
            self.element_pc_dict = pickle.load(f)

    @staticmethod
    def buckingham_potential(distance, a, b, c):
        warnings.filterwarnings("ignore")
        return a * numpy.exp(-b * distance) - c / numpy.power(distance, 6)

    def opt_r2_vals(self, constants, x_vals, y_vals, r2_minimize: bool):
        warnings.filterwarnings("ignore")
        y_space = self.buckingham_potential(x_vals, *constants)
        if numpy.all(numpy.isinf(y_space)) or numpy.all(numpy.isnan(y_space)) or numpy.any(numpy.isinf(y_space)):
            y_space[:] = 0
        r2 = r2_score(y_vals, y_space)
        return 1 - r2

    def get_master_frame(self):
        try:
            master_frame = pd.read_excel(self.master_file_uri, sheet_name=0)
            return master_frame.iloc[:, 1:]
        except BaseException:
            traceback.print_exc()

    def set_optimal_constants(self, distance, energy):
        opt_constants, opt_cov = curve_fit(self.buckingham_potential, distance, energy, self.init_constants)
        self.constants = opt_constants

    def gen_initial_values(self, x_vals, y_vals):
        maxX = max(x_vals)
        maxY = max(y_vals)
        maxXY = max(maxX, maxY)

        parameterBounds = [[-maxXY, maxXY], [-maxXY, maxXY], [-maxXY, maxXY]]

        result = differential_evolution(self.opt_r2_vals, parameterBounds, args=(x_vals, y_vals, True), seed=3)
        return result.x

    def get_element_pc_frame(self, element=None, partial_charge=None):
        master_frame = self.get_master_frame()
        if element is None and partial_charge is None:
            all_frames = []
            for ele in self.element_pc_dict:
                for char_index in range(len(self.element_pc_dict[ele])):
                    charge = float(self.element_pc_dict[ele][char_index])
                    ind_element_pc_frame = master_frame.loc[(master_frame['Element Name'] == ele) &
                                                            (master_frame['Partial Charge'] == charge)]
                    if len(ind_element_pc_frame) != 0:
                        all_frames.append(ind_element_pc_frame)
            return all_frames
        else:
            element_pc_frame = master_frame.loc[(master_frame['Element Name'] == element)
                                                & (master_frame['Partial Charge'] == partial_charge)]
            return element_pc_frame

    def evaluate_parameters(self, frame_to_eval: pd.DataFrame):
        x_vals = frame_to_eval['Distance (A)']
        y_vals = frame_to_eval['Relative LJ Enrgy']

        initial_constants = self.gen_initial_values(x_vals, y_vals)
        opt_constants, opt_cov = curve_fit(self.buckingham_potential, x_vals, y_vals, p0=self.init_constants,
                                           maxfev=10000)

        a_de, b_de, c_de = initial_constants[0], initial_constants[1], initial_constants[2]
        a_cv, b_cv, c_cv = opt_constants
        try:
            y_space_de = self.buckingham_potential(x_vals, a_de, b_de, c_de)
            y_space_cv = self.buckingham_potential(x_vals, a_cv, b_cv, c_cv)

            r2_de = r2_score(y_vals, y_space_de)
            r2_cv = r2_score(y_vals, y_space_cv)

            if r2_de >= r2_cv:
                return r2_de, a_de, b_de, c_de
            else:
                return r2_cv, a_cv, b_cv, c_cv

        except BaseException:
            opt_constants, opt_cov = curve_fit(self.buckingham_potential, x_vals, y_vals, p0=self.init_constants,
                                               maxfev=10000)
            a_cv, b_cv, c_cv = opt_constants
            y_space_cv = self.buckingham_potential(x_vals, a_cv, b_cv, c_cv)
            r2_cv = r2_score(y_vals, y_space_cv)
            return r2_cv, a_cv, b_cv, c_cv

    def set_eval_data(self, frame: pd.DataFrame, r2, a, b, c):
        data_row = [frame['Element Name'].iloc[0], frame['Atomic Mass'].iloc[0], frame['Partial Charge'].iloc[0],
                    r2, a, b, c]
        self.master_frame.loc[len(self.master_frame)] = data_row

    def set_r_squared_data(self, element=None, partial_charge=None):
        usable_frame = self.get_element_pc_frame(element, partial_charge)
        if element is None and partial_charge is None:
            for given_frame in usable_frame.copy():
                try:
                    r2, a, b, c = self.evaluate_parameters(given_frame)
                    self.set_eval_data(given_frame, r2, a, b, c)
                except BaseException:
                    traceback.print_exc()
                    r2, a, b, c = 0, 0, 0, 0
                    self.set_eval_data(given_frame, r2, a, b, c)
                    continue
        else:
            r2, a, b, c = self.evaluate_parameters(usable_frame)
            print('Value of R2 for the given element is', r2)
            print('Value of Constants for Buckingham potential equation for the given element is a = {}, b = {}, c = {}'
                  .format(a, b, c))

    def export_master_frame_to_sheet(self):
        self.master_frame.to_excel('final_data.xlsx')
