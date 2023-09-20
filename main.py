import numpy as np
from PIL import Image
from utils import plot_annotations
import argparse as ap
import os

from affine_rectification import affine_rectification
from utils import warp_image
import  matplotlib.pyplot as plt


DATA_CONFIG = {
        'data_dir' : './data/',
        'annotation' : './data/annotation',
        'q1' : {
            'images' : './data/q1',
            'annotation' : './data/annotation/q1_annotation.npy',
            },
        'q2' : {
            'images' : './data/q1',
            'annotation' : './data/annotation/q2_annotation.npy',
            },
        'q3' : {
            'images' : './data/q3',
            'annotation' : './data/annotation/q3_annotation.npy',
            }
        }


def main(args):
    config = DATA_CONFIG[args.question]

    images_dir = config['images']
    images = os.listdir(images_dir)
    annot_keys = list(map(lambda f: f.split('.')[0], images))
    image_files = dict([(a,b) for a,b in zip(annot_keys, images)])

    # filter images if specified
    if args.image is not None:
        try:
            image_files = {args.image: image_files[args.image]}
        except KeyError:
            print("Invalid filename!")
            return

    annotations = np.load(config['annotation'], allow_pickle=True)
    for image_id, image_file in image_files.items():
        image = Image.open(os.path.join(images_dir, image_file))
        img = np.array(image)
        annots = annotations.item().get(image_id)
        # plot_annotations(img, annots, plot_type=args.viz)
        H = affine_rectification(annots)
        res = warp_image(img, H)
        # plt.imshow(res)
        # plt.show()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3'], required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--image', default=None)
    parser.add_argument('-v', '--viz', default='lines')
    args = parser.parse_args()
    main(args)
