import itertools
from collections import Counter


# functions
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def create_all_possible_trafos(list_of_lists_of_params, identity_trafo):
    """
    Note: resulting param_combos list always starts with one identity trafo!
    :param list_of_lists_of_params, e.g., [[zoom_params], [rotation_params], [affine_params], [translation_params]]
    :param identify_trafo: trafo that does nothing
    :return: list of param combos
    """
    list_combos = list(powerset(range(len(list_of_lists_of_params))))
    id_trafo = identity_trafo
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


def put_trafo_params_into_patch_dict(list_of_params):
    trafo_list = [None] * len(list_of_params)
    for k in range(len(list_of_params)):
        param_combo = list_of_params[k]
        trafo_list[k] = {'rotate_degrees': param_combo[0],
                         'shift_UR_by': param_combo[1],
                         'shift_DP_by': param_combo[2],
                         'resize_UR_by': param_combo[3],
                         'resize_DP_by': param_combo[4]}
    return trafo_list


# augmentation
AUGM_LIST_OF_LISTS_OF_PARAMS = [[22.5, -22.5], [0.025, -0.025], [0.025, -0.025], [0.3], [None]]

AUGM_LIST_OF_PARAM_COMBOS = create_all_possible_trafos(list_of_lists_of_params=AUGM_LIST_OF_LISTS_OF_PARAMS,
                                                       identity_trafo=[0, 0, 0, 0, None])

AUGM_LIST_OF_DICTS = put_trafo_params_into_patch_dict(list_of_params=AUGM_LIST_OF_PARAM_COMBOS)

#
