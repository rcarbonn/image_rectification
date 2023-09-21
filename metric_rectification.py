import numpy as np
from perspective import gen_lines_and_intersection, proj_line, angle_change
from perspective import rectify_annots
from affine_rectification import affine_rectification



def metric_rectification(perp_annots):

    n_annots,_ = perp_annots.shape
    assert n_annots>=4, "Metric rectification needs atleast 2 perpendicular constraints"

    perp_annots_ = np.hstack((perp_annots, np.ones([n_annots,1], perp_annots.dtype)))
    perp_annots_ = np.split(perp_annots_, n_annots//2, axis=0)

    line_ids = [[0,1], [2,3]]
    angle_ids = [[4,5], [6,7]]

    A = np.zeros((2,3))
    for i,ids in enumerate(line_ids):
        l,m,p = gen_lines_and_intersection(perp_annots_[ids[0]], perp_annots_[ids[1]])
        A[i] = [l[0]*m[0], l[0]*m[1]+l[1]*m[0], l[1]*m[1]]

    _,_,s = np.linalg.svd(A)
    a,b,c = s[2]
    C = np.zeros((3,3))
    C[0][0] = a
    C[1][1] = c
    C[0][1] = b/2
    C[1][0] = b/2
    u,d,ut = np.linalg.svd(C)
    H = np.eye(3)
    H[0][0] = 1/np.sqrt(d[0])
    H[1][1] = 1/np.sqrt(d[1])
    H = H @ u.T
    print(H)
    return H

    # l_inf = proj_line(p_infs[0], p_infs[1])
    # data_type = l_inf.dtype

    # Ha = np.eye(3, dtype=data_type)
    # Hp = np.eye(3, dtype=data_type)
    # Hp[2,:] = l_inf
    # H = Ha@Hp
    # Hline = np.linalg.inv(H).T

    # # find angles between remaining lines
    # for ids in angle_ids:
        # la1,la2,_ = gen_lines_and_intersection(parallel_annots_[ids[0]], parallel_annots_[ids[1]])
        # angle_before, angle_after = angle_change(la1, la2, Hline)
        # print((angle_before, angle_after))

    # # rectified_annots = rectify_annots(parallel_annots[n_annots//2:].reshape(-1,1,2), H).squeeze(1)
    # rectified_annots = rectify_annots(parallel_annots.reshape(-1,1,2), H).squeeze(1)

    # return H, rectified_annots
