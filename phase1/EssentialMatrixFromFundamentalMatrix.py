import numpy as np

def essential_from_fundamental():
    """
    input: 
        intrinsics - 3 x 3
        fundamental_matrix - 3 x 3
    output:
        essential_matrix - 3 x 3
    """
    #estimate KT@F@K
    #take SVD of E
    #modify and reestimate E from (1,1,0) condition
    pass
