
import ra_utils
import ra_utils.utils

# import nnunetv2
# from pprint import pprint


def main():
    package_info_parameters = {
        "ra_utils": ra_utils.utils.package_infos(ra_utils),
        #"nnunetv2": cseg_utils.utils.package_infos(nnunetv2)
    }
    print("package_infos:::")
    pprint(package_info_parameters)
