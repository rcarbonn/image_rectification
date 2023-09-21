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
    print(Ht)

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    return result


def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0]**2 + u[1]**2) * np.sqrt(v[0]**2 + v[1]**2))


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


def plot_annotations(img, annots, plot_type='lines'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    n,_ = annots.shape
    colors = np.repeat(np.random.uniform(0, 1, (n//2,1,3)), 2, axis=0)
    if plot_type=='lines':
        ldata = np.split(annots.T, n//2, axis=1)
        for i,l in enumerate(ldata):
            line = l2d(l[0], l[1], color=colors[i], linewidth=5.0)
            ax.add_line(line)

    elif plot_type=='scatter':
        plt.scatter(annots[:,0], annots[:,1], c=colors)
    plt.show()
