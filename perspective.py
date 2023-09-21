import numpy as np
import cv2

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

def angle_change(l1, l2, Hline):
    l1_ = (Hline @ l1.reshape(-1,1)).flatten()[:2]
    l2_ = (Hline @ l2.reshape(-1,1)).flatten()[:2]
    l1 = l1[:2]
    l2 = l2[:2]
    angle_before = np.dot(l1, l2) / (np.linalg.norm(l1) * np.linalg.norm(l2))
    angle_after = np.dot(l1_, l2_) / (np.linalg.norm(l1_) * np.linalg.norm(l2_))
    return angle_before, angle_after

def rectify_annots(annots, H):
    rectified_annots = cv2.perspectiveTransform(annots, H)
    return rectified_annots
