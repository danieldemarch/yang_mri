import os
import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class BasePolicy(object):
    def __init__(self, args, model, env):
        self.args = args
        self.model = model
        self.env = env

        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build policy
            self._build_nets()
            self._build_ops()
            # initialize
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.args.exp_dir + '/summary')

    def save(self, filename='params'):
        fname = f'{self.args.exp_dir}/weights/{filename}.ckpt'
        self.saver.save(self.sess, fname)
        if self.args.finetune_model:
            fname = f'{self.args.exp_dir}/weights/model_{filename}.ckpt'
            self.model.saver.save(self.model.sess, fname)

    def load(self, filename='params'):
        fname = f'{self.args.exp_dir}/weights/{filename}.ckpt'
        self.saver.restore(self.sess, fname)
        if self.args.finetune_model:
            fname = f'{self.args.exp_dir}/weights/model_{filename}.ckpt'
            self.model.saver.restore(self.model.sess, fname)

    def scope_vars(self, scope, only_trainable=True):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf.get_collection(collection, scope=scope)
        assert len(variables) > 0
        logger.info(f"Variables in scope '{scope}':")
        for v in variables:
            logger.info("\t" + str(v))
        return variables

    def _build_nets(self):
        raise NotImplementedError()

    def _build_ops(self):
        raise NotImplementedError()

