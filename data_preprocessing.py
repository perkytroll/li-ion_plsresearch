import pandas as pd


class DataPreProcessing:
    def __init__(self):
        self.main_frame = pd.read_excel('final_data.xlsx', sheet_name=0)
        self.working_frame: pd.DataFrame = None
        self.normalized_working_frame: pd.DataFrame = None
        self.normalized_x_block: pd.DataFrame = None
        self.normalized_y_block: pd.DataFrame = None

    def standard_normalize(self) -> pd.DataFrame:
        mean_vector = []
        scaling_vector = []
        self.normalized_working_frame = self.working_frame.copy()
        for itr in range(self.normalized_working_frame.shape[1]):
            if self.normalized_working_frame.iloc[:, itr].dtype != object and \
                    self.normalized_working_frame.iloc[:, itr].name != 'R2':
                self.normalized_working_frame.iloc[:, itr] = pd.to_numeric(self.normalized_working_frame.iloc[:, itr])
                mean_vector.append(pd.Series.mean(pd.to_numeric(self.normalized_working_frame.iloc[:, itr]), axis=0))
                self.normalized_working_frame.iloc[:, itr] = self.normalized_working_frame.iloc[:, itr] - pd.Series.mean(
                    pd.to_numeric(self.normalized_working_frame.iloc[:, itr]), axis=0)
                scaling_vector.append(pd.Series.std(pd.to_numeric(self.normalized_working_frame.iloc[:, itr]), axis=0))
                self.normalized_working_frame.iloc[:, itr] = self.normalized_working_frame.iloc[:, itr] / pd.Series.std(
                    pd.to_numeric(self.normalized_working_frame.iloc[:, itr]), axis=0)
            else:
                pass
        print("Mean Vector is given by : \n", mean_vector)
        print("Scaling Vector is given by : \n", scaling_vector)

    def set_working_frame(self):
        self.working_frame = self.main_frame.copy()
        self.working_frame.drop('Element Name', axis=1, inplace=True)
        self.working_frame.drop('R2', axis=1, inplace=True)
        self.working_frame = self.working_frame.loc[:, ~self.working_frame.columns.str.contains('^Unnamed')]

    def divide_data_blocks(self):
        self.normalized_x_block = self.normalized_working_frame.copy()
        self.normalized_x_block.drop(['A', 'B', 'C'], axis=1, inplace=True)
        self.normalized_y_block = self.normalized_working_frame.copy()
        self.normalized_y_block.drop(['Atomic Mass', 'Partial Charge'], axis=1, inplace=True)

    def working_data_to_excel(self):
        self.working_frame.to_excel('working_data.xlsx')
