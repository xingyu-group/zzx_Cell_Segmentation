# requirement: install nibabel
import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

# only extract images from the training set.

# dir_path = '/home/zhaoxiang/Altas/Training/img'
# dir_path = '/home/zhaoxiang/Altas/Training/label'
dir_path = ''
filenames = os.listdir(dir_path)
filenames.sort()
for file in filenames:
    if file.startswith('.'):
        continue
    file_path = os.path.join(dir_path, file)
    img = nib.load(file_path)          # read nii
    img_fdata = img.get_fdata()
    fname = file.replace('.nii.gz', '')
    if fname.startswith('label'):
        fname = 'img' + fname[5:]
        img_f_path = os.path.join('/home/zhaoxiang/dataset/abdnomial_images', fname)
        raw_path = os.path.join(img_f_path, 'gt')
        a = 'gt'
    else:
        img_f_path = os.path.join('/home/zhaoxiang/dataset/abdnomial_images', fname)
        raw_path = os.path.join(img_f_path, 'raw')  
        a = 'raw' 
        
    if not os.path.exists(img_f_path):
        os.mkdir(img_f_path)
        
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)
        
        
    (x,y,z) = img.shape
    for i in range(z):
        gray = img_fdata[:,:,i]
        if np.max(gray)!=0:
            gray = np.where(gray == 6, 1, 0)
        matplotlib.image.imsave(os.path.join(raw_path, '{}_'.format(i) + a + '.png'), gray, cmap='gray')
    print(fname, '   done')
