from collections import OrderedDict
import itertools
from collections import Counter
import input.constants.class_init_constants as cic


# functions
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def create_all_possible_trafos(list_of_lists_of_params, identify_trafo):
    """
    Note: resulting param_combos list always starts with one identity trafo!
    :param list_of_lists_of_params, e.g., [[zoom_params], [rotation_params], [affine_params], [translation_params]]
    :param identify_trafo: trafo that does nothing
    :return: list of param combos
    """
    list_combos = list(powerset(range(len(list_of_lists_of_params))))
    id_trafo = identify_trafo
    param_combos = []
    for a in list_combos:
        list_combo_a = list(itertools.product(*[list_of_lists_of_params[i] for i in a]))
        # print(list_combo_a)
        for j in list_combo_a:
            # print(j)
            param_combo_j = id_trafo.copy()
            for i in range(len(a)):
                param_combo_j[a[i]] = j[i]
            # print(param_combo_j)
            param_combos = param_combos + [param_combo_j]

    # exclude none-unique combos (e.g. when a value given in list_of_Lists_of_params occurs in identity_trafo)
    param_combos = [p for p in Counter([tuple(n) for n in param_combos]).keys()]

    return param_combos


def put_trafo_params_into_patch_odict(list_of_params):
    trafo_list = [None] * len(list_of_params)
    for k in range(len(list_of_params)):
        param_combo = list_of_params[k]
        trafo_list[k] = OrderedDict([('rotate_degrees', param_combo[0]),
                                     ('shift_by', param_combo[1]),
                                     ('shift_along', param_combo[2]),
                                     ('resize_UR_by', param_combo[3]),
                                     ('resize_DP_by', param_combo[4])])
    return trafo_list


def put_trafo_params_into_whole_image_odict(list_of_params):
    trafo_list = [None] * len(list_of_params)
    for k in range(len(list_of_params)):
        param_combo = list_of_params[k]
        trafo_list[k] = OrderedDict([('trafo_odict',
                                      OrderedDict([('zoom', {'params': param_combo[0],
                                                             'units': None,
                                                             'resize_to_original': False,
                                                             'resize_with_padding': False}),
                                                   ('rotation', {'params': 0,
                                                                 'units': None,
                                                                 'resize_to_original': False,
                                                                 'resize_with_padding': False}),
                                                   ('affine', {'params': {'angle': param_combo[1],
                                                                          'shear': param_combo[2],
                                                                          'shift': param_combo[3]},
                                                               'units': None,
                                                               'resize_to_original': False,
                                                               'resize_with_padding': False}),
                                                   ('translation', {'params': [0, 0],
                                                                    'units': None,
                                                                    'resize_to_original': False,
                                                                    'resize_with_padding': False})
                                                   ])),
                                     ('relative_params_for_original_array', 'original'),
                                     ('final_resize_to_original', False),
                                     ('final_resize_with_padding', False)
                                     ])
    return trafo_list


# sibling augmentation
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_PROX = [[6, -6], [-0.015, -0.025], ['DP'], [0.1, -0.1], [None]]
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_DIST = [[6, -6], [0.015, 0.025], ['DP'], [0.1, -0.1], [None]]
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_PIP = [[6, -6], [0.025, -0.025], ['DP'], [0.1, -0.1], [None]]
"""
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_PROX = [[9, 6, 3, -3, -6, -9],
                                                 [0.005, -0.005, -0.015, -0.025],
                                                 ['DP'],
                                                 [0.15, 0.12, 0.09, 0.06, 0.03, -0.03, -0.06, -0.09, -0.12, -0.15],
                                                 [None]]
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_DIST = [[9, 6, 3, -3, -6, -9],
                                                 [-0.005, 0.005, 0.015, 0.025], ['DP'],
                                                 [0.15, 0.12, 0.09, 0.06, 0.03, -0.03, -0.06, -0.09, -0.12, -0.15],
                                                 [None]]
SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_PIP = [[9, 6, 3, -3, -6, -9],
                                            [0.025, 0.015, 0.005, -0.005, -0.015, -0.025], ['DP'],
                                            [0.15, 0.12, 0.09, 0.06, 0.03, -0.03, -0.06, -0.09, -0.12, -0.15],
                                            [None]]
"""
SIBLING_AUGM_LIST_OF_PARAM_COMBOS_MCP_PROX = \
    create_all_possible_trafos(list_of_lists_of_params=SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_PROX,
                               # [rotate_degrees, shift_by, shift_along, resize_UR_by, resize_DP_by]
                               identify_trafo=[0, 0, 'DP', 0, None])
SIBLING_AUGM_LIST_OF_PARAM_COMBOS_MCP_DIST = \
    create_all_possible_trafos(list_of_lists_of_params=SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_MCP_DIST,
                               # [rotate_degrees, shift_by, shift_along, resize_UR_by, resize_DP_by]
                               identify_trafo=[0, 0, 'DP', 0, None])
SIBLING_AUGM_LIST_OF_PARAM_COMBOS_PIP = \
    create_all_possible_trafos(list_of_lists_of_params=SIBLING_AUGM_LIST_OF_LISTS_OF_PARAMS_PIP,
                               # [rotate_degrees, shift_by, shift_along, resize_UR_by, resize_DP_by]
                               identify_trafo=[0, 0, 'DP', 0, None])

MOD_LIST_OF_ODICTS_MCP_PROX = \
    put_trafo_params_into_patch_odict(list_of_params=SIBLING_AUGM_LIST_OF_PARAM_COMBOS_MCP_PROX)
MOD_LIST_OF_ODICTS_MCP_DIST = \
    put_trafo_params_into_patch_odict(list_of_params=SIBLING_AUGM_LIST_OF_PARAM_COMBOS_MCP_DIST)
MOD_LIST_OF_ODICTS_PIP = \
    put_trafo_params_into_patch_odict(list_of_params=SIBLING_AUGM_LIST_OF_PARAM_COMBOS_PIP)

SIBL_AUGM_METHOD = 'augmentation'  # None
SIBL_AUGM_MODIFICATION_ORDER = ['rotate_patch', 'shift_patch', 'resize_patch']
SIBLING_AUGM_ODICT_OF_LIST_OF_ODICTS = OrderedDict([('SMP2', MOD_LIST_OF_ODICTS_MCP_PROX),
                                                    ('SMD2', MOD_LIST_OF_ODICTS_MCP_DIST),
                                                    ('SPP2', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SPD2', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SMP3', MOD_LIST_OF_ODICTS_MCP_PROX),
                                                    ('SMD3', MOD_LIST_OF_ODICTS_MCP_DIST),
                                                    ('SPP3', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SPD3', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SMP4', MOD_LIST_OF_ODICTS_MCP_PROX),
                                                    ('SMD4', MOD_LIST_OF_ODICTS_MCP_DIST),
                                                    ('SPP4', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SPD4', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SMP5', MOD_LIST_OF_ODICTS_MCP_PROX),
                                                    ('SMD5', MOD_LIST_OF_ODICTS_MCP_DIST),
                                                    ('SPP5', MOD_LIST_OF_ODICTS_PIP),
                                                    ('SPD5', MOD_LIST_OF_ODICTS_PIP)
                                                    ])

VAL_SIBL_AUGM_METHOD = None

# siblings
SIBLING_METHOD = None  # 'two_stream_zoom'
TWO_STREAM_MODS = [OrderedDict([('rotate_degrees', 0), ('shift_by', 0),
                                ('shift_along', 'DP'), ('resize_UR_by', 0), ('resize_DP_by', None)]),
                   OrderedDict([('rotate_degrees', 0), ('shift_by', 0),
                                ('shift_along', 'DP'), ('resize_UR_by', 0.5), ('resize_DP_by', None)])]

SIBLING_MODIFICATION_ORDER = ['rotate_patch', 'shift_patch', 'resize_patch']
SIBLING_ODICT_OF_LIST_OF_ODICTS = OrderedDict([('SMP2', TWO_STREAM_MODS),
                                               ('SMD2', TWO_STREAM_MODS),
                                               ('SPP2', TWO_STREAM_MODS),
                                               ('SPD2', TWO_STREAM_MODS),
                                               ('SMP3', TWO_STREAM_MODS),
                                               ('SMD3', TWO_STREAM_MODS),
                                               ('SPP3', TWO_STREAM_MODS),
                                               ('SPD3', TWO_STREAM_MODS),
                                               ('SMP4', TWO_STREAM_MODS),
                                               ('SMD4', TWO_STREAM_MODS),
                                               ('SPP4', TWO_STREAM_MODS),
                                               ('SPD4', TWO_STREAM_MODS),
                                               ('SMP5', TWO_STREAM_MODS),
                                               ('SMD5', TWO_STREAM_MODS),
                                               ('SPP5', TWO_STREAM_MODS),
                                               ('SPD5', TWO_STREAM_MODS)
                                               ])
VAL_SIBLING_METHOD = SIBLING_METHOD

# permutation
PERMUT_METHOD = None  # 'replace_each_sample'
PERMUT_SAMPLE_SIZE = 2
PERMUT_SAMPLE_N = 4
PERMUT_GROUP_PIC_SHAPE = 'square'

VAL_PERMUT_METHOD = None

# patch augmentation
PATCH_AUGM_METHOD = None
VAL_PATCH_AUGM_METHOD = None
"""[OrderedDict([('trafo_odict',
                  OrderedDict([('zoom', {'params': 1,
                                         'units': None,
                                         'resize_to_original': False,
                                         'resize_with_padding': False}),
                               ('rotation', {'params': 0,
                                             'units': None,
                                             'resize_to_original': False,
                                             'resize_with_padding': False}),
                               (
                                   'affine', {'params': {'angle': 0,
                                                         'shear': 0,
                                                         'shift': (
                                                             0, 0)},
                                              'units': None,
                                              'resize_to_original': False,
                                              'resize_with_padding': False}),
                               (
                                   'translation', {'params': [0, 0],
                                                   'units': None,
                                                   'resize_to_original': False,
                                                   'resize_with_padding': False})
                               ])),
                 ('relative_params_for_original_array', 'original'),
                 ('final_resize_to_original', False),
                 ('final_resize_with_padding', False)
                 ]),
    OrderedDict([('trafo_odict',
                  OrderedDict([('zoom', {'params': 1,
                                         'units': None,
                                         'resize_to_original': False,
                                         'resize_with_padding': False}),
                               ('rotation', {'params': 0,
                                             'units': None,
                                             'resize_to_original': False,
                                             'resize_with_padding': False}),
                               (
                                   'affine', {'params': {'angle': 0,
                                                         'shear': -0.3,
                                                         'shift': (
                                                             0, 0)},
                                              'units': None,
                                              'resize_to_original': False,
                                              'resize_with_padding': False}),
                               (
                                   'translation', {'params': [0, 0],
                                                   'units': None,
                                                   'resize_to_original': False,
                                                   'resize_with_padding': False})
                               ])),
                 ('relative_params_for_original_array', 'original'),
                 ('final_resize_to_original', False),
                 ('final_resize_with_padding', False)
                 ]),
    OrderedDict([('trafo_odict', OrderedDict([('zoom', {'params': 1.2,
                                                        'units': None,
                                                        'resize_to_original': False,
                                                        'resize_with_padding': False}),
                                              ('rotation', {'params': 10,
                                                            'units': None,
                                                            'resize_to_original': False,
                                                            'resize_with_padding': False}),
                                              (
                                                  'affine', {'params': {'angle': 0,
                                                                        'shear': 0,
                                                                        'shift': (
                                                                            0, 0)},
                                                             'units': None,
                                                             'resize_to_original': False,
                                                             'resize_with_padding': False}),
                                              (
                                                  'translation', {'params': [0, 0],
                                                                  'units': None,
                                                                  'resize_to_original': False,
                                                                  'resize_with_padding': False})
                                              ])),
                 ('relative_params_for_original_array', 'original'),
                 ('final_resize_to_original', False),
                 ('final_resize_with_padding', False)
                 ])]"""


# whole image augmentation
WHOLE_IMAGE_LIST_OF_LISTS_OF_PARAMS = [[1.1, 0.9],
                                       [10, -10],
                                       [0.2, -0.2],
                                       [(0.05, 0.05),
                                        (-0.05, 0.05),
                                        (0.05, -0.05),
                                        (-0.05, -0.05)]]

WHOLE_IMAGE_LIST_OF_PARAM_COMBOS = \
    create_all_possible_trafos(list_of_lists_of_params=WHOLE_IMAGE_LIST_OF_LISTS_OF_PARAMS,
                               # [zoom, rotation in degrees, shear, [(translation_y, translation_x)]
                               identify_trafo=[1, 0, 0, (0, 0)])
WHOLE_IMAGE_AUGM_METHOD = \
    put_trafo_params_into_whole_image_odict(list_of_params=WHOLE_IMAGE_LIST_OF_PARAM_COMBOS)

VAL_WHOLE_IMAGE_AUGM_METHOD = None

# debug patches
DEBUG_PATCHES_PARENT_PATCH_NAME_CHAIN_LISTS = OrderedDict([(i, [i]) for i in cic.ROI_NAMES])
                                              # OrderedDict([('SMP4', ['SMP4']),
                                              #              ('SMD4', ['SMD4']),
                                              #              ('SPP4', ['SPP4']),
                                              #              ('SPD4', ['SPD4'])])
DEBUG_PATCHES_PARENT_LEVEL = 'roi'
DEBUG_PATCHES_SUBLEVELS = 'all'
DEBUG_PATCHES_LEVELS = ['roi', 'sibling_augm', 'permut', 'sibling', 'patch_augm']
DEBUG_PATCHES_RANDOM_SUBSELECTION = False  # not yet implemented
DEBUG_PATCHES_SHOW_PLOT = False
DEBUG_PATCHES_SAVE_DIR = cic.HOME_DIR_SEP + \
                         '/autoscoRA/autoscoRA_Pipeline/output/scratch_output/augm_debug_figures'
DEBUG_PATCHES_SAVE_EXT = '.eps'
DEBUG_PATCHES_DPI = 100  # 1000


# debug whole images
DEBUG_WHOLE_IMAGES_SAVE_DIR = '/mnthome2/autoscoRA/autoscoRA_Pipeline/output/scratch_output/augm_debug_figures'
DEBUG_WHOLE_IMAGES_SHOW_PLOT = False
DEBUG_WHOLE_IMAGES_SAVE_EXT = '.eps'
DEBUG_WHOLE_IMAGES_DPI = 100


# final resize with padding
PATCH_RESIZE_WITH_PADDING = (64, 64)  # (224, 224)
IMAGE_RESIZE_WITH_PADDING = (256, 256)

#
