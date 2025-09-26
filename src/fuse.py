
import numpy as np
def fuse_probs(p_img, p_tab, alpha=0.6):
    return alpha*np.array(p_img)+(1-alpha)*np.array(p_tab)
