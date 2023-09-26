import numpy as np
from PIL import Image
from utils import plot_annotations
import argparse as ap
import os

from affine_rectification import affine_rectification
from metric_rectification import metric_rectification
from homography import get_homography
from perspective import rectify_annots, gen_metrics
from utils import warp_image, composite_image, gen_plots_q1, gen_plots_q2, gen_plots_q3, gen_eval_lines_plots, gen_fig
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


SUBMISSION_LIST = ['book1', 'checker1', 'chess1', 'tiles2', 'tiles4']

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
        fig1 = gen_fig(figsize=(15,7))
        fig1.suptitle("Affine rectification with 2 parallel annotations")
        fig_eval = gen_fig(figsize=(6,13))
        # fig_eval.suptitle("Test lines on input and rectified images")
        if args.question=='q2':
            fig2 = gen_fig(figsize=(15,7))
            fig2.suptitle("Metric rectification with 2 perpendicular annotations")
        for image_id, image_file in image_files.items():
            print(image_id)
            image = Image.open(os.path.join(images_dir, image_file))
            img = np.array(image)
            annots1 = annotations1.item().get(image_id)
            Haffine, Ha_line = affine_rectification(annots1)
            parallel_rectified_annots = rectify_annots(img, annots1, Haffine)
            eval_affine = gen_metrics(Ha_line, annots1, image_id)
            res = warp_image(img, Haffine)
            if args.question == 'q1' and image_id in SUBMISSION_LIST:
                fig1 = gen_plots_q1(fig1, count, img, annots1[:8], res)
                fig_eval = gen_eval_lines_plots(fig_eval, count, img, annots1[8:], res, parallel_rectified_annots[8:], eval_affine)
                count+=1
            if args.debug:
                plot_annotations(res, parallel_rectified_annots[8:], plot_type=args.viz)
            if args.question == 'q2':
                annotations2 = np.load(config['annotation2'], allow_pickle=True)
                annots2 = annotations2.item().get(image_id)
                perp_annots = rectify_annots(res, annots2, Haffine)
                Hmetric, Hm_line = metric_rectification(perp_annots)
                perp_rectified_annots = rectify_annots(res, perp_annots, Hmetric)
                res2 = warp_image(img, Hmetric@Haffine)
                eval_metric = gen_metrics(Hm_line@Ha_line, annots2, image_id)
                if args.debug:
                    plot_annotations(res, perp_annots[8:], plot_type=args.viz)
                    plot_annotations(res2, perp_rectified_annots[8:], plot_type=args.viz)
                if image_id in SUBMISSION_LIST:
                    fig2 = gen_plots_q2(fig2, count, img, annots2[:8], res, perp_annots[:8], res2)
                    fig_eval = gen_eval_lines_plots(fig_eval, count, img, annots2[8:], res2, perp_rectified_annots[8:], eval_metric)
                    count+=1
        n = plt.get_fignums()
        for i in n:
            plt.figure(i)
            plt.setp(plt.figure(i).get_axes(), xticks=[], yticks=[])
            plt.savefig("./results/%s_%d.png"%(args.question,i), bbox_inches="tight", dpi=300)
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
        fig = gen_fig()
        for i,src_file in enumerate(src_files):
            src_image = Image.open(os.path.join(images_dir, src_file))
            dst_image = Image.open(os.path.join(images_dir, dst_files[i]))
            src_img = np.array(src_image)
            if dst_img is None:
                dst_img = np.array(dst_image)
                save_img = dst_img.copy()
            h,w,c = src_img.shape
            src_pts = np.array([[0,0],[w,0],[w,h],[0,h]])
            dst_pts = annotations1.item().get(dst_ids[i])
            H = get_homography(src_pts, dst_pts)
            dst_img = composite_image(H, src_img, dst_img)

        if args.question=='q3':
            gen_plots_q3(fig, src_img, save_img, dst_pts, dst_img)
        elif args.question=='q5':
            plt.imshow(dst_img)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.tight_layout()
        plt.savefig("./results/%s_%d.png"%(args.question,1), bbox_inches="tight", dpi=300 )
        plt.show()




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3', 'q4', 'q5'], required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--image', default=None)
    parser.add_argument('-v', '--viz', default='lines')
    args = parser.parse_args()
    main(args)
