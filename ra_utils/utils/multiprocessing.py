
import torch 
import torch.multiprocessing as mp
import os


def set_multiprocessing_strategy():
    # https://github.com/Project-MONAI/MONAI/issues/701
    #   raise RuntimeError('received %d items of ancdata' %                                                                                                                                                                                                                                                                       
    #   RuntimeError: received 0 items of ancdata  
    # import resource
    # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    # Did not work ... 

    #
    #  Add this to submit script
    # export CSEG_UTILS_TORCH_MP_SHARING_STRATEGY="file_system"
    # export CSEG_UTILS_MP_METHOD="spawn"
    # export CSEG_UTILS_MP_METHOD="fork"
    # export CSEG_UTILS_MP_METHOD="forkserver"
    #
    # DANGER ZONE: I dont understand this. Best not use... 
    # Try different MP method to beat the error above
    CSEG_UTILS_MP_METHOD = os.environ.get('CSEG_UTILS_MP_METHOD')
    #print("CSEG_UTILS_MP_METHOD:", CSEG_UTILS_MP_METHOD)
    if CSEG_UTILS_MP_METHOD != None:  # 'spawn' or 'forkserver'

        mp.set_start_method(CSEG_UTILS_MP_METHOD, force=True)
        print(f"Setting mp.set_start_method to {CSEG_UTILS_MP_METHOD}")
    # Adjust sharing strategy
    # https://github.com/open-mmlab/mmdetection/issues/1520
    CSEG_UTILS_TORCH_MP_SHARING_STRATEGY = os.environ.get('CSEG_UTILS_TORCH_MP_SHARING_STRATEGY')
    if CSEG_UTILS_TORCH_MP_SHARING_STRATEGY != None: #  CSEG_UTILS_TORCH_MP_SHARING_STRATEGY = file_system 
        torch.multiprocessing.set_sharing_strategy(CSEG_UTILS_TORCH_MP_SHARING_STRATEGY)
        print(f"Setting torch.multiprocessing.set_sharing_strategy to {CSEG_UTILS_TORCH_MP_SHARING_STRATEGY}")