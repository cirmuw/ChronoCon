# TODO: later on, I can make the settings below a dict and include a loop in basic_framework.py
# that way, I can iterate through all joints/rois + different results saving dirs (so as not to overwrite)
# CAVE: if I start looping below the XRayList object creation, I cannot include "del xray_set" in the loop

# choose score type + score joint + roi
SCORE_TYPE = 'ero'  # 'ero'  # 'JSN'
ABS_OR_CHANGE = 'abs'
COMPARE_IMAGE_WITH_ITSELF = True
WHOLE_IMAGE_SCORE = False
# joint and ROI must be [ONE_JOINT] if WHOLE_IMAGE_SCORE == TRUE
# joint and ROI must be in same order and correspond to one another element-wise
# (but can have same ROI for several joints ("surjektiv") -> then need to list them twice/more times)
JOINT_DICT = {'MCP_ERO_P': ['MCPIIEP', 'MCPIIIEP', 'MCPIVEP', 'MCPVEP'],
              'MCP_ERO_D': ['MCPIIED', 'MCPIIIED', 'MCPIVED', 'MCPVED'],
              'PIP_ERO_P': ['PIPIIEP', 'PIPIIIEP', 'PIPIVEP', 'PIPVEP'],
              'PIP_ERO_D': ['PIPIIED', 'PIPIIIED', 'PIPIVED', 'PIPVED'],
              'MCP_JSN': ['MCPII', 'MCPIII', 'MCPIV', 'MCPV'],
              'PIP_JSN': ['PIPII', 'PIPIII', 'PIPIV', 'PIPV']}
ROI_DICT = {'SMP': ['SMP2', 'SMP3', 'SMP4', 'SMP5'],
            'SMD': ['SMD2', 'SMD3', 'SMD4', 'SMD5'],
            'SPP': ['SPP2', 'SPP3', 'SPP4', 'SPP5'],
            'SPD': ['SPD2', 'SPD3', 'SPD4', 'SPD5']}
JOINT_DICT_KEY = 'MCP_ERO_P'
ROI_DICT_KEY = 'SMP'
JOINT = JOINT_DICT[JOINT_DICT_KEY] + JOINT_DICT['MCP_ERO_D'] + JOINT_DICT['PIP_ERO_P'] + JOINT_DICT['PIP_ERO_D']
# ['PIPII']  # ['MCPIV']  # ['MCPIIEP', 'MCPIIIEP', 'MCPIVEP', 'MCPVEP']
ROI = ROI_DICT[ROI_DICT_KEY] + ROI_DICT['SMD'] + ROI_DICT['SPP'] + ROI_DICT['SPD']
# ['SPP2']  # ['SMD4']  # ['SMP2', 'SMP3', 'SMP4', 'SMP5']
# choose input type (change vs. absolute, whole image vs. patches)
PATCHES_OR_WHOLE_IMAGES = 'patches'  # 'whole_images'
SIBLINGS = True  # if siblings == False and len(siblings) > 1, only siblings[0] is used
# further options
LIST_TO_NUMPY = True

# random seed for shuffeling
RANDOM_SEED = 8905  # 8901

# data prep choices
USE_BALANCED_TRAIN_SET = False  # True
USE_BALANCED_VAL_SET = False  # False

# network choices
MODEL_TYPE = 'vgg16_manual'  # 'dense_net'  # 'vgg16'  # 'deep_simple'  # 'shallow_simple'
DATA_FORMAT = "channels_last"
BATCH_SIZE = 5
EPOCHS = 1

# saving results dir

#
