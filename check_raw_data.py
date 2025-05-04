import os
import glob
import numpy as np

PILOT_DIR = 'Pilot'

users = [d for d in os.listdir(PILOT_DIR) if os.path.isdir(os.path.join(PILOT_DIR, d))]

for user in sorted(users):
    user_dir = os.path.join(PILOT_DIR, user)
    print(f'用户: {user}')

    # 查找原始数据文件
    raw_files = glob.glob(os.path.join(user_dir, '*_fmcw_16bit_diff_profiles.npy'))
    print(f'  原始数据文件数量: {len(raw_files)}')
    if len(raw_files) == 0:
        print('  ⚠️ 缺失原始数据文件!')
    for f in raw_files:
        try:
            arr = np.load(f, mmap_mode='r')
            print(f'    {os.path.basename(f)}: shape={arr.shape}')
        except Exception as e:
            print(f'    {os.path.basename(f)}: 读取失败 ({e})')

    # 查找标注文件
    gt_files = glob.glob(os.path.join(user_dir, 'record_*_records.txt'))
    print(f'  标注文件数量: {len(gt_files)}')
    if len(gt_files) == 0:
        print('  ⚠️ 缺失标注文件!')
    for f in gt_files:
        print(f'    {os.path.basename(f)}')

    print('-' * 40) 