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
    # line_ids = [[4,5], [6,7]]
    angle_ids = [[4,5], [6,7]]

    A = np.zeros((4,3))
    for i,ids in enumerate(line_ids):
        l,m,p = gen_lines_and_intersection(perp_annots_[ids[0]], perp_annots_[ids[1]])
        A[i] = [l[0]*m[0], l[0]*m[1]+l[1]*m[0], l[1]*m[1]]

    _,_,s = np.linalg.svd(A)
    a,b,c = s[2]
    print(s[2])
    C = np.zeros((3,3))
    C[0][0] = a
    C[1][1] = c
    C[0][1] = b/2
    C[1][0] = b/2
    u,d,ut = np.linalg.svd(C)
    for i,ids in enumerate(line_ids):
        l,m,p = gen_lines_and_intersection(perp_annots_[ids[0]], perp_annots_[ids[1]])
        angs = l.reshape(1,-1) @ C @ m.reshape(-1,1)
    H = np.eye(3)
    H[0][0] = 1/np.sqrt(d[0])
    H[1][1] = 1/np.sqrt(d[1])
    H = H @ u.T
    Hline = np.linalg.inv(H).T

    return H,Hline
