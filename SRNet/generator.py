import numpy as np
from scipy import misc, io
from glob import glob
import random
from random import random as rand
from random import shuffle

def gen_flip_and_rot(cover_dir, stego_dir, thread_idx=0, n_threads=1):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['I_spatial']
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    
    iterable = zip(cover_list, stego_list)
    lists = list(iterable);
    while True:
        shuffle(lists) 
        for cover_path, stego_path in lists: 
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['I_spatial']
                batch[1,:,:,0] = io.loadmat(stego_path)['I_spatial']
            else:
                batch[0,:,:,0] = misc.imread(cover_path)
                batch[1,:,:,0] = misc.imread(stego_path)
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]
                              

def gen_valid(cover_dir, stego_dir, thread_idx=0, n_threads=1):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the stego directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the stego directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    load_mat=cover_list[0].endswith('.mat')
    if load_mat:
        img = io.loadmat(cover_list[0])['I_spatial']
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='float32')
    else:
        img = misc.imread(cover_list[0])
        img_shape = img.shape
        batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    img_shape = img.shape
    
    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            if  load_mat:
                batch[0,:,:,0] = io.loadmat(cover_path)['I_spatial']
                batch[1,:,:,0] = io.loadmat(stego_path)['I_spatial']
            else:
                batch[0,:,:,0] = misc.imread(cover_path)
                batch[1,:,:,0] = misc.imread(stego_path)
            yield [batch, labels]
