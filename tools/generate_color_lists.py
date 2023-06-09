import random

# def color_gen(num_classes=150):
lis = []
r_list = range(15, 240, 51)
g_list = range(16, 239, 40)
b_list = range(17, 238, 45)

if __name__ == '__main__':
    for i in r_list:
        for j in g_list:
            for k in b_list:
                lis.append([i + random.randint(-15, 15), j + random.randint(-15, 15), k + random.randint(-15, 15)])

print('Length of color list:', len(lis))
print('Color list: please copy this list to the conifg file of model. (palette=[...])')
print(lis)
