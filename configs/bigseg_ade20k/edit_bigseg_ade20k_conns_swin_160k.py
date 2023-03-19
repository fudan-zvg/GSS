_base_ = ['bigseg_ade20k_conns_swin_160k.py']

# edit_indice = [3812, 4950, 3615, 922, 6148, 3691, 1769, 4812, 4367, 6429, 6541, 369, 7497, 7039, 7654, 6470, 7180, 2057, 3074, 3079, 7459, 832, 6327, 3242, 2466, 2640, 943, 7551, 2458, 1925, 2210, 6868, 2867, 6519, 4547, 302, 778, 541, 7695, 1673, 357, 1581, 4194, 5166, 4734, 3963, 5143, 7415, 3571, 2013, 3769, 2118, 4634, 7196, 6877, 6422, 4668, 6778, 2761, 234, 5611, 1744, 2529, 4333, 3675, 302, 7736, 3592, 5373, 4435, 6816, 933, 1566, 3189, 5417, 7534, 1584, 1929, 6237, 3146, 563, 7976, 7064, 8176, 6353, 7436, 2632, 7622, 6248, 7590, 6120, 3557, 5972, 4333, 7158, 3698, 386, 4933, 1797, 4655, 751, 4080, 3171, 1797, 1855, 2643, 7518, 6023, 3177, 4589, 2735, 7493, 746, 165, 2196, 1336, 4806, 3628, 3560, 1862, 7677, 6264, 2574, 5546, 4205, 1788, 7353, 1282, 5834, 8009, 7430, 5433, 6963, 7963, 2448, 3567, 2253, 3218, 4717, 4810, 168, 6474, 6892, 394, 350, 6754, 4815, 3197, 1774, 5032]
edit_indice = [3812, 4950, 3615, 922, 6148, 3691, 1769, 4812, 4367, 6429, 6541, 369, 7497, 7039, 7654, 6470, 7180, 2057, 3074, 3079, 7459, 832, 6327, 3242, 2466, 2640, 943, 7551, 2458, 1925, 2210, 6868, 2867, 6519, 4547, 302, 778, 541, 7695, 1673, 357, 1581, 4194, 5166, 4734, 3963, 5143, 7415, 3571, 2013, 3769, 2118, 4634, 7196, 6877, 6422, 4668, 6778, 2761, 234, 5611, 1744, 2529, 4333, 3675, 302, 7736, 3592, 5373, 4435, 6816, 933, 1566, 3189, 5417, 7534, 1584, 1929, 6237, 3146, 563, 7976, 7064, 8176, 6353, 7436, 2632, 7622, 6248, 7590, 6120, 3557, 5972, 4333, 7158, 3698, 386, 4933, 1797, 4655, 751, 4080, 3171, 1797, 1855, 2643, 7518, 6023, 3177, 4589, 2735, 7493, 746, 165, 2196, 1336, 4806, 3628, 3560, 1862, 7677, 6264, 2574, 5546, 4205, 1788, 7353, 1282, 5834, 8009, 7430, 5433, 6963, 7963, 2448, 3567, 2253, 3218, 4717, 4810, 168, 6474, 6892, 394, 350, 6754, 4815, 3197, 1774, 5032, 1334]

model=dict(decode_head=dict(type='BigSegAggHeadRelaxE08HungerEdit', category_indice=edit_indice))

