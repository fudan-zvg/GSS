import os
# input_txt = '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/train.txt'
# output_txt = '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/train_mmseg.txt'

train_splits = {
    'ade20k-150-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train.txt',
    'bdd-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/bdd-relabeled/list/train.txt',
    'cityscapes-19-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/cityscapes-19-relabeled/list/train.txt',
    'coco-panoptic-133-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/train.txt',
    'idd-39-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/idd-39-relabeled/list/train.txt',
    'mapillary-public65-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/mapillary-public65-relabeled/list/train.txt',
    'sunrgbd-37-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/train.txt'
}
train_splits = list(train_splits.values())[:]

val_splits = {
    'ade20k-150-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/val.txt',
    'bdd-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/bdd-relabeled/list/val.txt',
    'cityscapes-19-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/cityscapes-19-relabeled/list/val.txt',
    'coco-panoptic-133-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/val.txt',
    'idd-39-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/idd-39-relabeled/list/val.txt',
    'mapillary-public65-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/mapillary-public65-relabeled/list/val.txt',
    'sunrgbd-37-relabeled': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/val.txt'
}
val_splits = list(val_splits.values())[:]

test_splits = {
    'camvid-11': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/camvid-11/list/val.txt',
    'kitti-19': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/kitti-19/list/val.txt',
    'pascal-context-60': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/pascal-context-60/list/val.txt',
    'scannet-20': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/scannet-20/list/val.txt',
    'voc2012': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/voc2012/list/val.txt',
    'wilddash-19': '/home/chenjiaqi/pj/mseg-api/mseg/dataset_lists/wilddash-19/list/val.txt'
}
test_splits = list(test_splits.values())[:]

if __name__ == '__main__':
    input_txt = test_splits[5]
    output_dir, file_name = os.path.split(input_txt)
    output_file_name = file_name.split('.')[0] + '_mmseg.txt'
    with open(os.path.join(output_dir, output_file_name), 'w') as out:
        with open(input_txt, 'r') as inp:
            line = inp.readline()
            print(line)
            while line:
                image_path = line.split(' ')[0]
                # xxx_dir = image_path.split('/')[-2]
                xxx = image_path.split('/')[-1].split('.')[0].split('_')[0]
                # xxx = image_path.split('.')[0]
                print(line, 'name:', xxx) # xxx_dir + '/'+
                out.write(xxx + '\n') # xxx_dir + '/'+
                line = inp.readline()





