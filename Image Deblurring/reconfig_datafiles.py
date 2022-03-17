import os
from shutil import copyfile

train_dir = 'GOPRO_Large/train/'
valid_dir = 'GOPRO_Large/valid/'

for i, path in enumerate([train_dir, valid_dir]):
    it = 0
    write_dir = 'valid' if i else 'train'
    for gp_dir in os.listdir(path):
        if gp_dir[0] == 'G':
            path_blur = path + gp_dir + '/blur/'
            path_sharp = path + gp_dir + '/sharp/'
            path_blur_gamma = path + gp_dir + '/blur_gamma/'
            for im_path in os.listdir(path_blur):
                copyfile(path_blur + im_path, f'{write_dir}_blur/{it}.png')
                copyfile(path_sharp + im_path, f'{write_dir}_sharp/{it}.png')
                copyfile(path_blur_gamma + im_path, f'{write_dir}_blur_gamma/{it}.png')
                it += 1

