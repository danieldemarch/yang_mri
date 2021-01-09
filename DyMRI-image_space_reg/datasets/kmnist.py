import os
import numpy as np
import torch
import tensorflow as tf

image_shape = [32, 32, 2]
data_path = './data/mnist/'
train_x, train_y = torch.load(data_path + 'training.pt')
train_x, train_y = train_x.numpy(), train_y.numpy()
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
train_x = train_x.astype(np.float32) / 255.
train_x = np.fft.fftshift(np.fft.fft2(train_x),axes=(-2,-1))
train_x = np.stack([np.real(train_x), np.imag(train_x)], axis=-1)
train_x = train_x.astype(np.float32)
test_x, test_y = torch.load(data_path + 'test.pt')
test_x, test_y = test_x.numpy(), test_y.numpy()
test_x = np.pad(test_x, ((0, 0), (2, 2), (2, 2)), mode='constant')
test_x = test_x.astype(np.float32) / 255.
test_x = np.fft.fftshift(np.fft.fft2(test_x),axes=(-2,-1))
test_x = np.stack([np.real(test_x), np.imag(test_x)], axis=-1)
test_x = test_x.astype(np.float32)


def sample_valid(x, y):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx], y[idx]

    x1, y1 = x[:-5000], y[:-5000]
    x2, y2 = x[-5000:], y[-5000:]

    return x1, y1, x2, y2

train_x, train_y, valid_x, valid_y = sample_valid(train_x, train_y)

# load mask
mask = np.load('../LOUPE/exp/kmnist/e0/mask_sample.npy')
m = np.repeat(np.repeat(mask[0], 32, axis=1), 2, axis=2)

def cartesian_mask(image):
    # mask = np.zeros_like(image)
    # N = np.random.randint(11)
    # idx = np.random.choice(image.shape[0], N, replace=False)
    # mask[idx] = 1

    # return mask
    return m

def _parse_train(i):
    image = train_x[i]
    label = train_y[i]        
    mask = cartesian_mask(image)

    return image, label, mask

def _parse_valid(i):
    image = valid_x[i]
    label = valid_y[i]
    mask = cartesian_mask(image)

    return image, label, mask

def _parse_test(i):
    image = test_x[i]
    label = test_y[i]
    mask = cartesian_mask(image)

    return image, label, mask

def get_dst(split):
    if split == 'train':
        size = train_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i],
                       [tf.float32, tf.int64, tf.float32])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = valid_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i],
                       [tf.float32, tf.int64, tf.float32])),
                      num_parallel_calls=16)
    else:
        size = test_x.shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i],
                       [tf.float32, tf.int64, tf.float32])),
                      num_parallel_calls=16)

    return dst, size


class Dataset(object):
    def __init__(self, split, batch_size):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, y, m = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size] + image_shape)
            self.y = tf.reshape(y, [batch_size])
            self.m = tf.reshape(m, [batch_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, y, m = self.sess.run([self.x, self.y, self.m])
        return {'x':x, 'y':y, 'm':m}

if __name__ == '__main__':
    dataset = Dataset('train', 100)
    dataset.initialize()
    batch = dataset.next_batch()
    print(batch)