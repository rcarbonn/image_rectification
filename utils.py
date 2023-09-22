from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as l2d
import numpy as np
import cv2


def normalize(v):
    return v / np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def warp_image(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return result


def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2))


def split_annotations(annots):
    n_annots,_ = annots.shape
    annots_ = np.hstack((annots, np.ones([n_annots,1], annots.dtype)))
    annots_ = np.split(annots_, n_annots//2, axis=0)
    annots = np.split(annots, n_annots//2, axis=0)
    return annots_, annots

def annotate(impath):
    im = Image.open(impath)
    im = np.array(im)

    clicks = []

    def click(event):
        x, y = event.xdata, event.ydata
        clicks.append([x, y, 1.])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(im)

    _ = fig.canvas.mpl_connect('button_press_event', click)
    plt.show()

    return clicks


def add_lines(ax, line_annots):
    n,_ = line_annots.shape
    ldata = np.split(line_annots.T, n//2, axis=1)
    colors = np.repeat(np.random.uniform(0, 1, (n//2,1,3)), 2, axis=0)
    for i,l in enumerate(ldata):
        line = l2d(l[0], l[1], color=colors[i], linewidth=5.0)
        ax.add_line(line)


def plot_annotations(img, annots, plot_type='lines'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    n,_ = annots.shape
    colors = np.repeat(np.random.uniform(0, 1, (n//2,1,3)), 2, axis=0)
    if plot_type=='lines':
        add_lines(ax, annots)

    elif plot_type=='scatter':
        plt.scatter(annots[:,0], annots[:,1], c=colors)
    plt.show()


def composite_image(H, src_img, dst_img):

    composite_img = dst_img.copy()
    warped_img = cv2.warpPerspective(src_img, H, (dst_img.shape[1], dst_img.shape[0]))
    mask = np.sum(warped_img,axis=-1) > 0
    composite_img[mask] = warped_img[mask]
    plt.imshow(composite_img)
    plt.show()


def gen_fig():
    return plt.figure()

def gen_plots(fig,idx, org_img, lines1, rect_img, rect_lines, rect_img2, eval_lines, rect_eval_lines):
    idx = (idx)*6
    print(idx)
    count=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(org_img)
    count+=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(org_img)
    add_lines(ax, lines1)
    count+=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(rect_img)
    add_lines(ax, rect_lines)
    count+=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(rect_img2)
    count+=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(org_img)
    add_lines(ax, eval_lines)
    count+=1

    ax = fig.add_subplot(6,6,idx+count)
    ax.imshow(rect_img2)
    add_lines(ax, rect_eval_lines)

    return fig


