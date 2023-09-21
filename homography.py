import numpy as np


def solve_dlt(x, x_):
    n,_ = x.shape
    x_ = np.expand_dims(x_,axis=1)
    cross_ = np.cross(x_, np.identity(3)*-1)
    print(cross_.shape)
    a = []
    for i in range(n):
        r = np.zeros((3,9))
        r[0,:3] = x[i]
        r[1,3:6] = x[i]
        r[2,6:] = x[i]
        p = cross_[i] @ r
        a.append(p[0])
        a.append(p[1])
    A = np.array(a)
    _,_,vh = np.linalg.svd(A)
    h = vh[-1]
    H = h.reshape(3,3)
    return H/H[-1,-1]




def get_homography(src_pts, dst_pts):

    assert src_pts.shape == dst_pts.shape, "src_pts and dst_pts should have equal number of points"
    n_pts,_ = src_pts.shape
    src_pts_ = np.hstack((src_pts, np.ones([n_pts,1])))
    dst_pts_ = np.hstack((dst_pts, np.ones([n_pts,1])))

    # normalize src and dst pts
    xmean = np.mean(src_pts, axis=0)
    xmean_ = np.mean(dst_pts, axis=0)
    x = src_pts - xmean
    x_ = dst_pts - xmean_
    sx = np.sqrt(2)/np.max(np.linalg.norm(x, axis=1))
    sx_ = np.sqrt(2)/np.max(np.linalg.norm(x_, axis=1))
    T = np.array([[sx, 0, -sx*xmean[0]],
                 [0, sx, -sx*xmean[1]],
                 [0, 0, 1]])
    T_ = np.array([[sx_, 0, -sx_*xmean_[0]],
                 [0, sx_, -sx_*xmean_[1]],
                 [0, 0, 1]])
    xnorm = T@src_pts_.T
    xnorm_ = T_@dst_pts_.T

    # DLT
    Hbar = solve_dlt(xnorm.T, xnorm_.T)
    H = np.linalg.inv(T_) @ Hbar @ T
    H = H/H[-1,-1]
    print(H)
    return H

    # iterative optimization
