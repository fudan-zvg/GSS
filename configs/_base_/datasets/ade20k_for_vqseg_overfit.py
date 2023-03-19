_base_ = [
    'ade20k_for_vqseg.py'
]

data = dict(
    train=dict(
        img_dir='images/validation',
        ann_dir='annotations/validation'))