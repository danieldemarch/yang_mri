import os
import numpy as np
import scipy.sparse as sparse
import pickle
import h5py
import tensorflow as tf

image_shape = [128, 128, 2]
data_path = './data/fastMRI/knee/'
train_folder = data_path+'singlecoil_train/'
train_files = os.listdir(train_folder)[:-50]
train_files = [train_folder+f for f in train_files]
valid_files = os.listdir(train_folder)[-50:]
valid_files = [train_folder+f for f in valid_files]
test_folder = data_path+'singlecoil_val/'
test_files = os.listdir(test_folder)
test_files = [test_folder+f for f in test_files]

def crop(x):
    return x[96:-96, 96:-96]

def to_kspace(x):
    x = np.fft.fftshift(np.fft.fft2(x),axes=(-2,-1))
    x = np.stack([np.real(x), np.imag(x)], axis=-1)
    x = x.astype(np.float32)
    return x

def cartesian_mask(image, max_acquisition, center_acquisition):
    mask = np.zeros_like(image)
    H = image.shape[0]
    N = np.random.randint(max_acquisition-center_acquisition+1)
    pad = (H - center_acquisition + 1) // 2
    center_idx = range(pad, pad+center_acquisition)
    choices = list(set(range(H)) - set(center_idx))
    idx = np.random.choice(choices, N, replace=False)
    mask[center_idx] = 1
    mask[idx] = 1
    return mask

def load_sparse_data(files):
    data = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            x = hf['reconstruction_esc'][:]
        num_slices = x.shape[0]
        for xs in x[num_slices//4:3*num_slices//4]:
            xs = crop(xs)
            data.append(sparse.coo_matrix(xs))
    return data

with open(data_path+'preprocessed_data_cache.pkl', 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)

def _parse_train(i, max_acquisition, center_acquisition):
    image = train_data[i].todense()
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = to_kspace(image)
    mask = cartesian_mask(image, max_acquisition, center_acquisition)
    return image, mask

def _parse_valid(i, max_acquisition, center_acquisition):
    image = valid_data[i].todense()
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = to_kspace(image)
    mask = cartesian_mask(image, max_acquisition, center_acquisition)
    return image, mask

def _parse_test(i, max_acquisition, center_acquisition):
    image = test_data[i].todense()
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = to_kspace(image)
    mask = cartesian_mask(image, max_acquisition, center_acquisition)
    return image, mask

def get_dst(split, max_acquisition, center_acquisition):
    if split == 'train':
        size = len(train_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, max_acquisition, center_acquisition],
                       [tf.float32, tf.float32])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = len(valid_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, max_acquisition, center_acquisition],
                       [tf.float32, tf.float32])),
                      num_parallel_calls=16)
    else:
        size = len(test_data)
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, max_acquisition, center_acquisition],
                       [tf.float32, tf.float32])),
                      num_parallel_calls=16)

    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, max_acquisition, center_acquisition):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, max_acquisition, center_acquisition)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, m = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size] + image_shape)
            self.m = tf.reshape(m, [batch_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, m = self.sess.run([self.x, self.m])
        return {'x':x, 'm':m}

if __name__ == '__main__':
    dataset = Dataset('train', 100, 32, 16)
    dataset.initialize()
    batch = dataset.next_batch()
    print(np.min(batch['x']), np.mean(batch['x']), np.max(batch['x']))