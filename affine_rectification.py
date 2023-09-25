import numpy as np
from perspective import gen_lines_and_intersection, proj_line, angle_change
from perspective import rectify_annots

def affine_rectification(parallel_annots):

    n_annots,_ = parallel_annots.shape
    assert n_annots>=4, "Affine rectification needs atleast 2 parallel constraints"

    parallel_annots_ = np.hstack((parallel_annots, np.ones([n_annots,1], parallel_annots.dtype)))
    parallel_annots_ = np.split(parallel_annots_, n_annots//2, axis=0)

    line_ids = [[0,1], [2,3]]
    angle_ids = [[4,5], [6,7]]

    p_infs = []
    for ids in line_ids:
        l1,l2,p = gen_lines_and_intersection(parallel_annots_[ids[0]], parallel_annots_[ids[1]])
        p_infs.append(p)

    l_inf = proj_line(p_infs[0], p_infs[1])
    data_type = l_inf.dtype

    Ha = np.eye(3, dtype=data_type)
    Hp = np.eye(3, dtype=data_type)
    Hp[2,:] = l_inf
    H = Ha@Hp
    Hline = np.linalg.inv(H).T

    return H, Hline
