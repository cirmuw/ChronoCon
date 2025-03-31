# gt roi
GT_TYPE = None  # 'handsegm_handloc'
GT_SQUARE = False
GT_SIZE_METHOD = 'MC_PP_MP_bone_multiple_CMC'
GT_SIZE_SCALE = 1
GT_MCP_PROX_SCALE = 0.5
GT_MCP_DIST_SCALE = 0.49
GT_PIP_PROX_SCALE = 0.5
GT_PIP_DIST_SCALE = 0.5
GT_MCP_PROX_CENTER_SHIFT = 0.14/GT_MCP_PROX_SCALE
GT_MCP_DIST_CENTER_SHIFT = 0.05/GT_MCP_DIST_SCALE
GT_PIP_PROX_CENTER_SHIFT = 0/GT_PIP_PROX_SCALE
GT_PIP_DIST_CENTER_SHIFT = 0/GT_PIP_DIST_SCALE
GT_UR_DP_LENGTH = 'DP'
GT_UR_DP_ORIENTATION = 'DP'

# predicted roi
PRED_TYPE = 'autosegm_autoloc'
PRED_SQUARE = False
PRED_SIZE_METHOD = 'MC_PP_MP_bone_multiple_CMC'
PRED_SIZE_SCALE = 1
PRED_MCP_PROX_SCALE = 0.5
PRED_MCP_DIST_SCALE = 0.49
PRED_PIP_PROX_SCALE = 0.5
PRED_PIP_DIST_SCALE = 0.5
# abs_shift = median * m_shift = length_DP * shift --> shift = median/length_DP * m_shift = m_shift/scale
PRED_MCP_PROX_CENTER_SHIFT = 0.14/PRED_MCP_PROX_SCALE
PRED_MCP_DIST_CENTER_SHIFT = 0.05/PRED_MCP_DIST_SCALE
PRED_PIP_PROX_CENTER_SHIFT = 0/PRED_PIP_PROX_SCALE
PRED_PIP_DIST_CENTER_SHIFT = 0/PRED_PIP_DIST_SCALE
PRED_UR_DP_LENGTH = 'DP'
PRED_UR_DP_ORIENTATION = 'DP'

# patch cutting
CUT_SQUARE = False
RESIZE_PATCH = None
PADD_PATCH = None
BASE_CROP = 3

# segm outliers
OUTLIER_THRESH = {'IoU_odict': [0.8, 'smaller'], 'Dice_odict': [0.8, 'smaller'],
                  'length_UR_abs_rel_errors_odict': [0.1, 'greater'],
                  'length_DP_abs_rel_errors_odict': [0.1, 'greater'],
                  'length_UR_abs_errors_odict': [4, 'greater'],
                  'length_DP_abs_errors_odict': [4, 'greater'],
                  'orientation_UR_abs_errors_odict': [5, 'greater'],
                  'center_errors_odict': [4, 'greater']}

#
