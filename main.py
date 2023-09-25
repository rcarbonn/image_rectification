import numpy as np
from PIL import Image
from utils import plot_annotations
import argparse as ap
import os

from affine_rectification import affine_rectification
from metric_rectification import metric_rectification
from homography import get_homography
from perspective import rectify_annots, gen_metrics
from utils import warp_image, composite_image, gen_plots, gen_fig
import matplotlib.pyplot as plt



DATA_CONFIG = {
        'data_dir' : './data/',
        'annotation' : './data/annotation',
        'q1' : {
            'images' : './data/q1',
            'annotation1' : './data/annotation/q1_annotation2.npy',
            },
        'q2' : {
            'images' : './data/q1',
            'annotation1' : './data/annotation/q1_annotation2.npy',
            'annotation2' : './data/annotation/q2_annotation2.npy',
            },
        'q3' : {
            'images' : './data/q3',
            'annotation1' : './data/annotation/q3_annotation2.npy',
            },
        'q5' : {
            'images' : './data/q5',
            'annotation1' : './data/annotation/q3_annotation2.npy',
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

    annotations1 = np.load(config['annotation1'], allow_pickle=True)
    if args.question == 'q1' or args.question == 'q2':
        count = 0
        fig = gen_fig()
        for image_id, image_file in image_files.items():
            print(image_id)
            image = Image.open(os.path.join(images_dir, image_file))
            img = np.array(image)
            annots1 = annotations1.item().get(image_id)
            Haffine, Ha_line = affine_rectification(annots1)
            parallel_rectified_annots = rectify_annots(img, annots1, Haffine)
            print(parallel_rectified_annots)
            gen_metrics(Ha_line, annots1, 'test')
            res = warp_image(img, Haffine)
            plot_annotations(res, parallel_rectified_annots[8:], plot_type=args.viz)
            if args.question == 'q2':
                annotations2 = np.load(config['annotation2'], allow_pickle=True)
                annots2 = annotations2.item().get(image_id)
                perp_annots = rectify_annots(res, annots2, Haffine)
                Hmetric, Hm_line = metric_rectification(perp_annots)
                perp_rectified_annots = rectify_annots(res, perp_annots, Hmetric)
                res2 = warp_image(img, Hmetric@Haffine)
                gen_metrics(Hm_line@Ha_line, annots2, 'test')
                plot_annotations(res, perp_annots[8:], plot_type=args.viz)
                plot_annotations(res2, perp_rectified_annots[8:], plot_type=args.viz)
                # fig = gen_plots(fig, count, img, annots2[:8], res, perp_annots[:8], res2, annots2[8:], perp_rectified_annots[8:])
            count+=1
        # plot_annotations(res, rectified_annots, plot_type=args.viz)
        # plt.imshow(res2)
        plt.show()
    elif args.question == 'q3' or args.question == 'q5':
        src_files = []
        dst_files = []
        dst_ids = []
        for image_id, image_file in image_files.items():
            if 'normal' in image_id:
                src_files.append(image_file)
            elif 'perspective' in image_id:
                dst_files.append(image_file)
                dst_ids.append(image_id)
        Hlist = []
        dst_img = None
        for i,src_file in enumerate(src_files):
            src_image = Image.open(os.path.join(images_dir, src_file))
            dst_image = Image.open(os.path.join(images_dir, dst_files[i]))
            src_img = np.array(src_image)
            if dst_img is None:
                dst_img = np.array(dst_image)
            h,w,c = src_img.shape
            src_pts = np.array([[0,0],[w,0],[w,h],[0,h]])
            dst_pts = annotations1.item().get(dst_ids[i])
            H = get_homography(src_pts, dst_pts)
            dst_img = composite_image(H, src_img, dst_img)




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3', 'q4', 'q5'], required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--image', default=None)
    parser.add_argument('-v', '--viz', default='lines')
    args = parser.parse_args()
    main(args)
