import os
import shutil
import numpy as np


def filelist_generation():
    # dataset = 'FODB'
    dataset = 'fivek'
    data_root = 'data/%s/' % dataset
    save_path = 'data/'
    category_list = sorted(os.listdir(data_root))

    img_list = []
    # Each item in img_list should contain: (path of image, text label, numerical label)
    # For example: (/home/bob/Editprint/data/FODB/orig/D01_img_orig_0042.jpg, orig, 0)
    for idx, category in enumerate(category_list):
        category_path = data_root + category + '/'
        category_imgs = sorted(os.listdir(category_path))
        np.random.shuffle(category_imgs)
        for img_path in category_imgs:
            img_list.append((category_path + img_path, category, idx))

    print('#Image: %d' % len(img_list))
    print('#Category: %d' % len(category_list))
    save_name = '%s_img%d_cat%d.txt' % (dataset, len(img_list), len(category_list))
    textfile = open(save_path + save_name, 'w')
    for item in img_list:
        textfile.write('%s$@%s$@%s\n' % (item[0], item[1], item[2]))  # $@ is delimiter
    textfile.close()


if __name__ == '__main__':
    filelist_generation()
