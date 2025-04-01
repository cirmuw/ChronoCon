import input.constants.input_constants as const
import os
import re


nested_joint_list = [
    ### HANDS ###
    [["IPIEP", "PIPIIEP", "PIPIIIEP", "PIPIVEP", "PIPVEP"], "ERO", "H"],  # 0
    [["IPIED", "PIPIIED", "PIPIIIED", "PIPIVED", "PIPVED"], "ERO", "H"],  # 1
    [["PIPII", "PIPIII", "PIPIV", "PIPV"], "JSN", "H"],                   # 2
    [["MCPIEP", "MCPIIEP", "MCPIIIEP", "MCPIVEP", "MCPVEP"], "ERO", "H"], # 3
    [["MCPIED", "MCPIIED", "MCPIIIED", "MCPIVED", "MCPVED"], "ERO", "H"], # 4
    [["MCPI", "MCPII", "MCPIII", "MCPIV", "MCPV"], "JSN", "H"],           # 5
    [["Base_MCIE"], "ERO", "H"],                                          # 6
    [["CMCIII", "CMCIV", "CMCV"], "JSN", "H"],                            # 7
    [["LunatE", "RadiusE", "ScaphE", "TrapE", "UlnaE"], "ERO", "H"],      # 8
    [["LunatE"], "ERO", "H"],                                             # 9
    [["RadiusE"], "ERO", "H"],                                            # 10
    [["ScaphE"], "ERO", "H"],                                             # 11
    [["TrapE"], "ERO", "H"],                                              # 12
    [["UlnaE"], "ERO", "H"],                                              # 13
    [["Rad_Carp", "Sca_Cap", "Tra_Sca"], "JSN", "H"],                     # 14
    [["Rad_Carp"], "JSN", "H"],                                           # 15
    [["Tra_Sca"], "JSN", "H"],                                            # 16
    [["Sca_Cap"], "JSN", "H"],                                            # 17
    ### FEET ###
    [["MTPIEP"], "ERO", "F"],                                             # 18
    [["MTPIED"], "ERO", "F"],                                             # 19
    [["MTPIIEP", "MTPIIIEP", "MTPIVEP", "MTPVEP"], "ERO", "F"],           # 20
    [["MTPIIED", "MTPIIIED", "MTPIVED", "MTPVED"], "ERO", "F"],           # 21
    [["IPEP"], "ERO", "F"],                                               # 22
    [["IPED"], "ERO", "F"],                                               # 23
    [["MTPI"], "JSN", "F"],                                               # 24
    [["MTPII", "MTPIII", "MTPIV", "MTPV"], "JSN", "F"],                   # 25
    [["IP"], "JSN", "F"]                                                  # 26
]

nested_name_list = [
    ### HANDS ###
    ["PIP_I-V_EP", "ERO", "H"],                                           # 0
    ["PIP_I-V_ED", "ERO", "H"],                                           # 1
    ["PIP_II-V", "JSN", "H"],                                             # 2
    ["MCP_I-V_EP", "ERO", "H"],                                           # 3
    ["MCP_I-V_ED", "ERO", "H"],                                           # 4
    ["MCP_I-V", "JSN", "H"],                                              # 5
    ["Base_MCIE", "ERO", "H"],                                            # 6
    ["CMC_III-V", "JSN", "H"],                                            # 7
    ["SUM_-_LunatE,RadiusE,ScaphE,TrapE,UlnaE", "ERO", "H"],              # 8
    ["LunatE", "ERO", "H"],                                               # 9
    ["RadiusE", "ERO", "H"],                                              # 10
    ["ScaphE", "ERO", "H"],                                               # 11
    ["TrapE", "ERO", "H"],                                                # 12
    ["UlnaE", "ERO", "H"],                                                # 13
    ["SUM_-_Rad_Carp,Sca_Cap,Tra_Sca", "JSN", "H"],                       # 14
    ["Rad_Carp", "JSN", "H"],                                             # 15
    ["Tra_Sca", "JSN", "H"],                                              # 16
    ["Sca_Cap", "JSN", "H"],                                              # 17
    ### FEET ###
    ["MTP_I_EP", "ERO", "F"],                                             # 18
    ["MTP_I_ED", "ERO", "F"],                                             # 19
    ["MTP_II-V_EP", "ERO", "F"],                                          # 20
    ["MTP_II-V_ED", "ERO", "F"],                                          # 21
    ["IP_EP", "ERO", "F"],                                                # 22
    ["IP_ED", "ERO", "F"],                                                # 23
    ["MTP_I", "JSN", "F"],                                                # 24
    ["MTP_II-V", "JSN", "F"],                                             # 25
    ["IP", "JSN", "F"]                                                    # 26
]


def rename_pauls_stupid_encoding(directory, code_list=nested_name_list, rename=False, renamed=True):

    if renamed:
        path_list = [i for i in os.listdir(directory) if not i.startswith('.')]
        prefix_list = []
        for p in path_list:
            prefix = re.sub("_.*", "", p)
            prefix_list = prefix_list + [prefix]
    else:
        path_list = [i for i in os.listdir(directory) if 'joint' in i and not i.startswith('.')]
        prefix_list = []
        for p in path_list:
            suffix = re.sub(".*_joint", "joint", p)
            extension = os.path.splitext(suffix)[1]
            suffix = os.path.splitext(suffix)[0]
            suffix_nr = int(re.sub("joint_", "", suffix))
            prefix = re.sub("_joint.*", "", p)
            prefix = re.sub("\\.", "-", prefix)
            prefix = re.sub("_", "", prefix)
            prefix_list = prefix_list + [prefix]

            suffix_txt = "_".join(list(reversed(code_list[suffix_nr])))

            new_path = prefix + "_" + suffix_txt + extension

            if rename:
                print(p, "  vs  ", new_path)
                os.rename(directory + os.sep + p, directory + os.sep + new_path)

    return list(set(prefix_list))


if __name__ == "__main__":
    directory = const.HOME_DIR_SEP + "autoscoRA/autoscoRA_Pipeline/evaluation/final_scoring_results"
    # directory = const.HOME_DIR_SEP + "autoscoRA/autoscoRA_Pipeline/evaluation/model_selection/GridSearch3_hand"
    # directory = const.HOME_DIR_SEP + "autoscoRA/autoscoRA_Pipeline/evaluation/model_selection/GridSearch3_foot"
    rename_pauls_stupid_encoding(code_list=nested_name_list, directory=directory, rename=True, renamed=False)
"""    for i in [j for j in os.listdir(directory) if not j.startswith('.')]:
        print(i)
        new_name = directory + os.sep + re.sub('TESTGridSearch29', 'FINALRESULTS', i)
        old_name = directory + os.sep + i
        os.rename(old_name, new_name)"""
#
