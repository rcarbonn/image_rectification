import numpy as np

def proj_line(pt1, pt2):
    l = np.cross(pt1, pt2)
    return l/l[-1]

def intersection_pt(l1, l2):
    pt = np.cross(l1, l2)
    return pt/pt[-1]

def gen_lines_and_intersection(pts1, pts2):
    # get a line between pts1 and another line between pts2 and intersection among them
    l1 = proj_line(pts1[0], pts1[1])
    l2 = proj_line(pts2[0], pts2[1])
    p = intersection_pt(l1, l2)
    return l1,l2,p


def affine_rectification(parallel_annots):

    n_annots,_ = parallel_annots.shape
    assert n_annots>=4, "Affine rectification needs atleast 2 parallel constraints"

    parallel_annots_ = np.hstack((parallel_annots, np.ones([n_annots,1], parallel_annots.dtype)))
    parallel_annots_ = np.split(parallel_annots_, n_annots//2, axis=0)

    #choose first two lines, add random selection
    idx1 = 0; idx2 = 1
    l1,l2,p1 = gen_lines_and_intersection(parallel_annots_[idx1], parallel_annots_[idx2])

    idx3 = 2; idx4 = 3
    l3,l4,p2 = gen_lines_and_intersection(parallel_annots_[idx3], parallel_annots_[idx4])

    l_inf = proj_line(p1, p2)
    data_type = l_inf.dtype

    Ha = np.eye(3, dtype=data_type)
    Hp = np.eye(3, dtype=data_type)
    Hp[2,:] = l_inf
    H = Ha@Hp
    H_line = np.linalg.inv(H).T

    # find angles between remaining lines
    angle_ids = [[4,5], [6,7]]
    # angle_ids = [[0,1], [2,3]]
    for ids in angle_ids:
        la1,la2,_ = gen_lines_and_intersection(parallel_annots_[ids[0]], parallel_annots_[ids[1]])
        la1_ = (H_line @ la1.reshape(-1,1)).flatten()
        la2_ = (H_line @ la2.reshape(-1,1)).flatten()
        la1 = la1[:2]
        la2 = la2[:2]
        angle_before = np.dot(la1, la2) / (np.linalg.norm(la1) * np.linalg.norm(la2))
        la1_ = la1_[:2]
        la2_ = la2_[:2]
        angle_after = np.dot(la1_, la2_) / (np.linalg.norm(la1_) * np.linalg.norm(la2_))
        print((angle_before, angle_after))

    return H
