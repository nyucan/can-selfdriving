from PIL import Image
import os.path
from glob import glob

label_dir_list = ['./data/ori/selected-output-0910', './data/ori/selected-output-0910_1']
data_dir_list = ['./data/ori/image-0910', './data/ori/image-0910_1']

im_id = 1

for i in range(len(label_dir_list)):
    label_dir, data_dir = label_dir_list[i], data_dir_list[i]
    label_paths = os.listdir(label_dir)
    image_paths = os.listdir(data_dir)
    for j in range(len(label_paths)):
        label_file = Image.open(label_dir + '/' + label_paths[j])
        label_file = label_file.convert(mode='RGB')
        label_file = label_file.crop((0, 0, 160, 48))
        label_file.save('./data/training/gt_image_2/' + str(im_id) + '.png')

        img_name = label_paths[j][:-4]

        data_file = Image.open(data_dir + '/' + img_name + '.jpeg')
        data_file = data_file.convert(mode='RGB')
        data_file = data_file.crop((0, 0, 160, 48))
        data_file.save('./data/training/image_2/' + str(im_id) + '.png')

        im_id += 1
