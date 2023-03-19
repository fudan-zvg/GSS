from torchvision.utils import save_image, make_grid
import torch
import os

dir = '/SSD_DISK/users/chenjiaqi/data/nyudepthv2/train/student_lounge_0001/01871.h5'

if __name__ == '__main__':
    import h5py
    with h5py.File(dir, "r") as f:
        rgb = torch.tensor(f['rgb'])
        depth = torch.tensor(f['depth'])
        save_image(rgb / rgb.max(), '../work_dirs/depth_show/rgb_' + os.path.split(dir)[-1].split('.')[0] + '.png')
        save_image((depth), '../work_dirs/depth_show/depth_' + os.path.split(dir)[-1].split('.')[0] + '.png')

        # for key in f.keys():
        #     # print(f[key], key, f[key].name, f[key].value)
        #     # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
        #     print(f[key])
        #     print(key)
        #     print(f[key])

import matplotlib.pyplot as plt
plt.figure()
plt.imshow()