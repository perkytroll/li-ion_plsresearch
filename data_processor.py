import pickle
import traceback
from pathlib import Path, PosixPath
import pandas as pd
import re
from mendeleev import element
from collections import defaultdict


class RawDataCreation:
    def __init__(self):
        self.master_frame = pd.DataFrame()
        self.sim_results = list(Path("./sim_data/").rglob("*.[xX][lL][sS][xX]"))
        self.element_pc_dict = defaultdict(list)

    @staticmethod
    def get_element_name(file_path: PosixPath) -> str:
        element_name = file_path.__str__().split('/')[1]
        return element_name

    @staticmethod
    def get_atomic_mass(element_name: str):
        si = element(element_name)
        return si.atomic_weight

    @staticmethod
    def get_partial_charge(file_path: PosixPath) -> str:
        init_pc = file_path.__str__().split('/')[2]
        exact_pc = re.search('([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?', init_pc)
        return exact_pc.group(1)

    def create_master_excel(self):
        for file in self.sim_results:
            try:
                ele_data_frame = pd.read_excel(file, sheet_name=0)

                element_name = self.get_element_name(file_path=file)
                ele_data_frame['Element Name'] = element_name

                partial_charge = self.get_partial_charge(file_path=file)
                ele_data_frame['Partial Charge'] = partial_charge

                atomic_mass = self.get_atomic_mass(element_name)
                ele_data_frame['Atomic Mass'] = atomic_mass

                self.element_pc_dict[element_name].append(partial_charge)
                self.master_frame = pd.concat([self.master_frame, ele_data_frame], ignore_index=True)
            except BaseException:
                traceback.print_exc()

        with open('element_pc_dict.pkl', 'wb') as f:
            pickle.dump(self.element_pc_dict, f)
        self.master_frame.to_excel('raw_data.xlsx')
