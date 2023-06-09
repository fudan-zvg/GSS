import os
import argparse
import torch
import random

def parse_args():
    parser = argparse.ArgumentParser(description='GSS posterior learning.')
    parser.add_argument('--dataset', help='specify the dataset name', default='ade20k')
    args = parser.parse_args()
    return args


def main(args):
    if args.dataset == 'ade20k':
        lis = []
        r_list = range(15, 240, 51)
        g_list = range(16, 239, 40)
        b_list = range(17, 238, 45)
        for i in r_list:
            for j in g_list:
                for k in b_list:
                    lis.append([i + random.randint(-15, 15), j + random.randint(-15, 15), k + random.randint(-15, 15)])
        print('-----------  ADE20K  -----------')
        print('Length of color list:', len(lis))
        print('-----------  Begin  ------------')
        print('Color list: please copy this list to the conifg file of model. (palette=[...])')
        print(lis)
        print('------------- End --------------')
    elif args.dataset == 'mseg':
        lis = []
        r_list = range(15, 255, 55)
        g_list = range(16, 255, 51)
        b_list = range(17, 255, 33)

        if __name__ == '__main__':
            for i in r_list:
                for j in g_list:
                    for k in b_list:
                        lis.append([i + random.randint(-15, 0), j + random.randint(-15, 0), k + random.randint(-15, 0)])

        print('-----------  ADE20K  -----------')
        print('Length of color list:', len(lis[6:]))
        print('-----------  Begin  ------------')
        print('Color list: please copy this list to the conifg file of model. (palette=[...])')
        print(lis[6:])
        print('------------- End --------------')
    else:
        raise NotImplementedError()
    print('You can manually use DALL-E pretrained VQVAE to reconstruct maskige '
          'and eventually find that some categories have relatively low IoU (Intersection over Union) values. '
          'Generally, these categories have low IoU because their colors are too gray (RGB values close to [128, 128, 128]) '
          'and not vibrant enough. '
          'You can manually modify them to a more vivid color, such as changing them to [0, 255, 0], '
          'and continuously adjust the colors of the low IoU categories. '
          'This can further improve the mIoU (mean Intersection over Union) of Stage 1.')


if __name__ == '__main__':
    args = parse_args()
    main(args)