from data_preprocessing import DataPreProcessing
from data_processor import RawDataCreation
from data_visualizer import DataVisualizer
from function_to_data import FunctionToData
from nipals import NIPALS
from pca import PCA

# rdc = RawDataCreation()
# rdc.create_master_excel()
#
# func = FunctionToData()
# func.set_r_squared_data()
# func.export_master_frame_to_sheet()

pca = PCA()
pca.set_working_frame()
pca.standard_normalize()
pca.data_in_score_space(pca.normalized_working_frame, num_components=1, num_of_comps=True)
pca.summary_statistics(pca.normalized_working_frame)

pca.data_in_score_space(pca.normalized_working_frame, num_components=2, num_of_comps=True)
pca.summary_statistics(pca.normalized_working_frame)

pca.plot_score_space()

# nipals = NIPALS()
# nipals.set_working_frame()
# nipals.working_data_to_excel()
# nipals.standard_normalize()
# nipals.divide_data_blocks()
# nipals.set_x_y_blocks()
#
# dv = DataVisualizer()
# dv.plot_x_block_variables(nipals.normalized_x_block)
# nipals.nipals_iteration_implementation()
