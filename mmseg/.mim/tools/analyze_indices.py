import scipy.io as sio
import torch
import glob
import numpy

def get_freq(indices):
    freq = torch.zeros(8192)
    indices = indices.flatten()
    for indice in indices:
        freq[indice] += 1
    return freq

# # 统计所有indice出现的频次，直方图
root_path = 'work_dirs/anal/val'
list = glob.glob(root_path + '/*')
total = torch.zeros(8192)
for item in list:
    data = sio.loadmat(item)
    freq = get_freq(data['vq_indice_gt'])
    print('sum', freq.sum())
    freq[freq != 0] = 1
    print('zeros', freq.sum())
    print()
    # total += freq
# sio.savemat('work_dirs/anal/val_gt_freq_1024x2048.mat', {'data': total.cpu().numpy()})
# count = total.clone()
# count[count <= 50] = 0
# count[count > 50] = 1
# print(count.sum().item())

data = sio.loadmat('work_dirs/anal/val_gt_freq_1024x2048.mat')
freq_dict = dict()
for i in range(len(data['data'][0])):
    freq_dict[i] = data['data'][0][i]
sorted_freq_dict = sorted(freq_dict.items(), reverse=True, key=lambda x: x[1])

# 这个list相当于从logist到8192的映射，只需要sorted_freq_list[logist]就可以得到真实的预测值
sorted_freq_list = list(dict(sorted_freq_dict).keys())

# 这个list相当于从8192到logist范围的映射，直接inv_freq_list[gt]就可以得到所有gt对应的新indice，把所有大于最大值的都ignore掉即可
inv_freq_list = torch.zeros(8192)
for i in range(8192):
    inv_freq_list[sorted_freq_list[i]] = i

sio.savemat('ckp/t_dicts.mat', {'t_full2quant_dict': inv_freq_list.numpy(), 't_quant2full_dict': sorted_freq_list})
data = sio.loadmat('ckp/t_dicts.mat')
print(data)

if __name__ == '__main__':
    print('ok')
