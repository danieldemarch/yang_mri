import logging
import numpy as np
import tensorflow as tf

from datasets import get_dataset

logger = logging.getLogger()

class RecEnv(object):
    def __init__(self, args, split):
        self.args = args
        self.dataset = get_dataset(split, args)
        self.state_size = self.dataset.image_shape
        self.action_size = self.dataset.image_shape[0]+1
        self.terminal_act = self.action_size-1
        if hasattr(self.dataset, 'cost'):
            self.cost = self.dataset.cost
        else:
            self.cost = np.array([self.args.acquisition_cost] * self.action_size, dtype=np.float32)
        self.dataset.initialize()

    def reset(self, loop=True, init=False):
        if init:
            self.dataset.initialize()
        try:
            batch = self.dataset.next_batch()
            self.x = batch['x']
            self.m = np.zeros_like(self.x)
            return self.x * self.m, self.m.copy()
        except:
            if loop:
                self.dataset.initialize()
                batch = self.dataset.next_batch()
                self.x = batch['x']
                self.m = np.zeros_like(self.x)
                return self.x * self.m, self.m.copy()
            else:
                return None, None
    
    def step(self, action, prediction):
        empty = action == -1
        terminal = action == self.terminal_act
        normal = np.logical_and(~empty, ~terminal)
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        done = np.zeros([action.shape[0]], dtype=np.bool)
        if np.any(empty):
            done[empty] = True
            reward[empty] = 0.
        if np.any(terminal):
            done[terminal] = True
            p = prediction[terminal]
            x = self.x[terminal]
            x = x[...,0] + x[...,1] * 1j
            x = np.expand_dims(np.absolute(np.fft.ifft2(np.fft.ifftshift(x,axes=(-2,-1)))),axis=-1)
            m = self.m[terminal]
            mse = np.sum(np.square(p-x), axis=tuple(range(1,x.ndim)))
            num = np.sum(1-m, axis=tuple(range(1,x.ndim))) + 1e-8
            reward[terminal] = -mse / num
        if np.any(normal):
            m = self.m[normal]
            a = action[normal]
            assert np.all(m[np.arange(len(a)), a] == 0)
            m[np.arange(len(a)), a] = 1.
            self.m[normal] = m.copy()
            reward[normal] = -self.cost[a]
        
        return self.x * self.m, self.m.copy(), reward, done 

