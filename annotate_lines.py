import numpy as np
import argparse as ap

from utils import annotate

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
            }
        }


def mark_lines(args):
    config = DATA_CONFIG[args.question]
    annotations1 = np.load(config['annotation1'], allow_pickle=True)
    if args.question == 'q1':
        print("Annotate parallel lines")
        lines = annotate(args.image_path)
        lines = np.array(lines)[:,:2]
        annot_id = args.image_path.split('/')[-1].split('.')[0]
        annotations1.item()[annot_id] = lines
        np.save('data/annotation/q1_annotation2', annotations1)
    if args.question == 'q2':
        print("Annotate perpendicular lines")
        annotations2 = np.load(config['annotation2'], allow_pickle=True)
        lines = annotate(args.image_path)
        lines = np.array(lines)[:,:2]
        annot_id = args.image_path.split('/')[-1].split('.')[0]
        annotations2.item()[annot_id] = lines
        np.save('data/annotation/q2_annotation2', annotations2)
    if args.question == 'q3':
        print("Annotate 4 points")
        annotations2 = np.load(config['annotation1'], allow_pickle=True)
        pts = annotate(args.image_path)
        pts = np.array(pts)[:,:2]
        annot_id = args.image_path.split('/')[-1].split('.')[0]
        annotations2.item()[annot_id] = pts
        np.save('data/annotation/q3_annotation2', annotations2)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-q', '--question', choices=['q1', 'q2', 'q3'], required=True)
    parser.add_argument('-i', '--image_path', default=None, required=True)
    args = parser.parse_args()
    mark_lines(args)
