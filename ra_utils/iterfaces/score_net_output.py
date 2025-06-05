

import torch
from typing import Literal

# def score_output(y: torch.tensor,
#                  option: Literal["None", "MHClassifer2Reg"] = "None"):
#     if option == "None":
#         return y    
#     elif option == "MHClassifer2Reg":
#         return y[:,0]  # instead of class logits this should be the regression output
#     else: 
#         raise ValueError(f"Interface option score_output :: {option = } not supported ")