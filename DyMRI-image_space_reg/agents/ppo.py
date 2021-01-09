import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple, defaultdict

from utils.memory import ReplayMemory
from utils.visualize import plot_dict
from utils.nets import convnet, dense_nn
from utils.metrics import compute_ssim

from .base import BasePolicy

logger = logging.getLogger()

class PPOPolicy(BasePolicy):
    def __init__(self, args, model, env):
        super(PPOPolicy, self).__init__(args, model, env)

    def act(self, state, mask, auxilary, hard=False):
        probas = self.sess.run(self.actor_proba,
                              {self.state:state,
                               self.mask:mask,
                               self.auxilary:auxilary})
        if hard:
            action = np.array([np.argmax(p) for p in probas])
        else:
            action = np.array([np.random.choice(self.env.action_size, p=p) for p in probas])

        return action

    def to_image(self, x):
        x = tf.complex(x[...,0], x[...,1])
        x = tf.signal.ifftshift(x, axes=(-2,-1))
        x = tf.signal.ifft2d(x)
        x = tf.abs(x)
        x = tf.expand_dims(x, axis=-1)

        return x

    def _build_nets(self):
        state_size = self.env.state_size
        action_size = self.env.action_size
        auxilary_size = state_size[:-1] + [self.model.auxilary_dim]
        self.state = tf.placeholder(tf.float32, shape=[None]+state_size, name='state')
        self.mask = tf.placeholder(tf.float32, shape=[None]+state_size, name='mask')
        self.auxilary = tf.placeholder(tf.float32, shape=[None]+auxilary_size, name='auxilary')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')

        self.old_logp_a = tf.placeholder(tf.float32, shape=[None], name='old_logp_a')
        self.v_target = tf.placeholder(tf.float32, shape=[None], name='v_target')
        self.adv = tf.placeholder(tf.float32, shape=[None], name='advantage')

        with tf.variable_scope('embedding'):
            embed = tf.concat([self.to_image(self.state), self.mask, self.auxilary], axis=-1)
            embed = convnet(embed, self.args.embed_layers, name='embed')
            self.embed_vars = self.scope_vars('embedding')

        with tf.variable_scope('actor'):
            actor_layers = self.args.actor_layers + [action_size]
            self.actor = dense_nn(embed, actor_layers, name='actor')
            logits_mask = self.mask[:, :, 0, 0]
            if hasattr(self.env, 'terminal_act'):
                logits_mask = tf.concat([logits_mask, tf.zeros([tf.shape(logits_mask)[0],1])], axis=1)
            inf_tensor = -tf.ones_like(self.actor) * np.inf
            self.actor_logits = tf.where(tf.equal(logits_mask, 0), self.actor, inf_tensor)
            self.actor_proba = tf.nn.softmax(self.actor_logits)
            self.actor_log_proba = tf.nn.log_softmax(self.actor_logits)
            self.actor_entropy = tf.distributions.Categorical(probs=self.actor_proba).entropy()
            index = tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1)
            self.logp_a = tf.gather_nd(self.actor_log_proba, index)
            self.actor_vars = self.scope_vars('actor')

        with tf.variable_scope('critic'):
            critic_layers = self.args.critic_layers + [1]
            self.critic = tf.squeeze(dense_nn(embed, critic_layers, name='critic'))
            self.critic_vars = self.scope_vars('critic')

    def _build_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.clip_range = tf.placeholder(tf.float32, shape=None, name='ratio_clip_range')

        with tf.variable_scope('actor_train'):
            ratio = tf.exp(self.logp_a - self.old_logp_a)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_a = - tf.reduce_mean(tf.minimum(self.adv * ratio, self.adv * ratio_clipped))
            if self.args.ent_coef > 0:
                loss_a -= tf.reduce_mean(self.actor_entropy) * self.args.ent_coef

            optim_a = tf.train.AdamOptimizer(self.lr_a)
            grads_and_vars = optim_a.compute_gradients(loss_a, var_list=self.actor_vars+self.embed_vars)
            grads_a, vars_a = zip(*grads_and_vars)
            if self.args.clip_grad_norm > 0:
                grads_a, gnorm_a = tf.clip_by_global_norm(grads_a, clip_norm=self.args.clip_grad_norm)
                gnorm_a = tf.check_numerics(gnorm_a, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_a', gnorm_a)
            grads_and_vars = zip(grads_a, vars_a)
            self.train_op_a = optim_a.apply_gradients(grads_and_vars)

        with tf.variable_scope('critic_train'):
            loss_c = tf.reduce_mean(tf.square(self.v_target - self.critic))

            optim_c = tf.train.AdamOptimizer(self.lr_c)
            grads_and_vars = optim_c.compute_gradients(loss_c, var_list=self.critic_vars+self.embed_vars)
            grads_c, vars_c = zip(*grads_and_vars)
            if self.args.clip_grad_norm > 0:
                grads_c, gnorm_c = tf.clip_by_global_norm(grads_c, clip_norm=self.args.clip_grad_norm)
                gnorm_c = tf.check_numerics(gnorm_c, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_c', gnorm_c)
            grads_and_vars = zip(grads_c, vars_c)
            self.train_op_c = optim_c.apply_gradients(grads_and_vars)

        self.train_ops = tf.group(self.train_op_a, self.train_op_c)

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')

            self.summary = [
                tf.summary.scalar('loss/adv', tf.reduce_mean(self.adv)),
                tf.summary.scalar('loss/ratio', tf.reduce_mean(ratio)),
                tf.summary.scalar('loss/loss_actor', loss_a),
                tf.summary.scalar('loss/loss_critic', loss_c),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]

            self.summary += [tf.summary.histogram('vars/' + v.name, v)
                            for v in vars_a if v is not None]
            self.summary += [tf.summary.histogram('vars/' + v.name, v)
                            for v in vars_c if v is not None]

            self.summary += [tf.summary.scalar('grads/' + g.name, tf.norm(g))
                            for g in grads_a if g is not None]
            self.summary += [tf.summary.scalar('grads/' + g.name, tf.norm(g))
                            for g in grads_c if g is not None]

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
            
    def _generate_rollout(self, buffer):
        s, m = self.env.reset()
        obs = []
        masks = []
        futures = []
        actions = []
        rewards = []
        flags = []
        episode_reward = np.zeros([s.shape[0]], dtype=np.float32)

        logger.info('start rollout.')
        done = np.zeros([s.shape[0]], dtype=np.bool)
        while not np.all(done):
            f = self.model.auxilary(s, m)
            a_orig = self.act(s, m, f)
            a = a_orig.copy()
            a[done] = -1
            p = self.model.predict(s, m)
            s_next, m_next, re, done = self.env.step(a, p)
            ri = self.model.reward(s, m, a, s_next, m_next, done)
            r = re + ri
            obs.append(s)
            masks.append(m)
            futures.append(f)
            actions.append(a_orig)
            rewards.append(r)
            flags.append(done)
            episode_reward += r
            s, m = s_next, m_next
        logger.info('rollout finished.')

        # length of the episode.
        T = len(rewards)

        # compute the current log pi(a|s) and predicted v values.
        with self.sess.as_default():
            if False:
                logp_a = self.logp_a.eval({self.action: np.concatenate(actions), 
                                        self.state: np.concatenate(obs),
                                        self.mask: np.concatenate(masks),
                                        self.auxilary: np.concatenate(futures)})
                logp_a = logp_a.reshape([T, -1])
                assert not np.any(np.isnan(logp_a)), 'logp_a contains NaN values.'
                assert not np.any(np.isinf(logp_a)), 'logp_a contains Inf values.'

                v_pred = self.critic.eval({self.state: np.concatenate(obs),
                                        self.mask: np.concatenate(masks),
                                        self.auxilary: np.concatenate(futures)})
                v_pred = v_pred.reshape([T, -1])
                assert not np.any(np.isnan(v_pred)), 'v_pred contains NaN values.'
                assert not np.any(np.isinf(v_pred)), 'v_pred contains Inf values.'
            else:
                logp_a_list = []
                v_pred_list = []
                for at, xt, mt, ft in zip(actions, obs, masks, futures):
                    logp_a = self.logp_a.eval({self.action: at,
                                               self.state: xt,
                                               self.mask: mt,
                                               self.auxilary: ft})
                    logp_a_list.append(logp_a)
                    assert not np.any(np.isnan(logp_a)), 'logp_a contains NaN values.'
                    assert not np.any(np.isinf(logp_a)), 'logp_a contains Inf values.'

                    v_pred = self.critic.eval({self.state: xt,
                                               self.mask: mt,
                                               self.auxilary: ft})
                    v_pred_list.append(v_pred)
                    assert not np.any(np.isnan(v_pred)), 'v_pred contains NaN values.'
                    assert not np.any(np.isinf(v_pred)), 'v_pred contains Inf values.'
                logp_a = np.stack(logp_a_list)
                v_pred = np.stack(v_pred_list)

        # record this batch
        logger.info('record this batch.')
        n_rec = 0
        x = self.env.x.copy()
        for i in range(s.shape[0]):
            done = [f[i] for f in flags]
            max_T = np.min(np.where(done)[0])
            n_rec += max_T
            state = [s[i] for s in obs][:max_T+1]
            mask = [m[i] for m in masks][:max_T+1]
            future = [f[i] for f in futures][:max_T+1]
            action = [a[i] for a in actions][:max_T+1]
            reward = [r[i] for r in rewards][:max_T+1]
            logp = logp_a[:max_T+1, i]
            vp = v_pred[:max_T+1, i]
            next_state = s_next[i]
            next_mask = m_next[i]

            # Compute TD errors
            td_errors = [reward[t] + self.args.gamma * vp[t + 1] - vp[t] for t in range(max_T)]
            td_errors += [reward[max_T] + self.args.gamma * 0.0 - vp[max_T]]  # handle the terminal state.

            # Estimate advantage backwards.
            advs = []
            adv_so_far = 0.0
            for delta in td_errors[::-1]:
                adv_so_far = delta + self.args.gamma * self.args.lam * adv_so_far
                advs.append(adv_so_far)
            advs = advs[::-1]
            assert len(advs) == max_T+1

            # Estimate critic target
            vt = np.array(advs) + np.array(vp)

            # add into the memory buffer
            for t, (s, m, f, a, old_logp_a, v_target, adv) in enumerate(zip(
                state, mask, future, action, logp, vt, advs)):
                buffer.add(buffer.tuple_class(x[i], s, m, f, a, old_logp_a, v_target, adv))
        logger.info(f'record done: {n_rec} transitions added.')

        return np.mean(episode_reward), n_rec

    def _ratio_clip_fn(self, n_iter):
        clip = self.args.ratio_clip_range
        if self.args.ratio_clip_decay:
            delta = clip / self.args.iters
            clip -= delta * n_iter

        return max(0.0, clip)
    
    def run(self):
        BufferRecord = namedtuple('BufferRecord', ['x', 's', 'm', 'f', 'a', 'old_logp_a', 'v_target', 'adv'])
        buffer = ReplayMemory(tuple_class=BufferRecord, capacity=self.args.buffer_size)

        reward_history = []
        reward_averaged = []
        best_reward = -np.inf
        step = 0
        total_rec = 0
        
        for n_iter in range(self.args.iters):
            clip = self._ratio_clip_fn(n_iter)
            if self.args.clean_buffer: buffer.clean()
            ep_reward, n_rec = self._generate_rollout(buffer)
            reward_history.append(ep_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            total_rec += n_rec

            for batch in buffer.loop(self.args.record_size, self.args.epochs):
                if self.args.finetune_model and n_iter >= self.args.finetune_warmup:
                    self.model.finetune(batch)

                _, summ_str = self.sess.run(
                    [self.train_ops, self.merged_summary],
                    feed_dict={self.lr_a: self.args.lr_a,
                               self.lr_c: self.args.lr_c,
                               self.clip_range: clip,
                               self.state: batch['s'],
                               self.mask: batch['m'],
                               self.auxilary: batch['f'],
                               self.action: batch['a'],
                               self.old_logp_a: batch['old_logp_a'],
                               self.v_target: batch['v_target'],
                               self.adv: batch['adv'],
                               self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,
                               }
                )
                self.writer.add_summary(summ_str, step)
                step += 1
            
            if self.args.log_freq > 0 and (n_iter+1) % self.args.log_freq == 0:
                logger.info("[iteration:{}/step:{}], best:{}, avg:{:.2f}, clip:{:.2f}; {} transitions.".format(
                    n_iter, step, np.max(reward_history), np.mean(reward_history[-10:]), clip, total_rec))

            if self.args.eval_freq > 0 and n_iter % self.args.eval_freq == 0:
                self.evaluate(folder=f'{n_iter}', load=False)

            if self.args.save_freq > 0 and (n_iter+1) % self.args.save_freq == 0:
                self.save()

            if np.mean(reward_history[-10:]) > best_reward:
                best_reward = np.mean(reward_history[-10:])
                self.save('best')

        # FINISH
        self.save()
        logger.info("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))
        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_dict(f'{self.args.exp_dir}/learning_curve.png', data_dict, xlabel='episode')

    def evaluate(self, folder='test', load=True, max_batches=10):
        if load: self.load('best')
        metrics = defaultdict(list)
        transitions = []
        init = True
        num_batches = 0
        while True:
            num_batches += 1
            s, m = self.env.reset(loop=False, init=init)
            init = False
            if s is None or m is None:
                break
            if num_batches > max_batches:
                break
            num_acquisition = np.sum(m[:,:,0,0], axis=1)
            episode_reward = np.zeros([s.shape[0]], dtype=np.float32)
            transition = m.copy()
            done = np.zeros([s.shape[0]], dtype=np.bool)
            while not np.all(done):
                f = self.model.auxilary(s, m)
                a = self.act(s, m, f, hard=True)
                a[done] = -1
                p = self.model.predict(s, m)
                s_next, m_next, re, done = self.env.step(a, p)
                ri = self.model.reward(s, m, a, s_next, m_next, done)
                r = re + ri
                episode_reward += r
                num_acquisition += ~done
                s, m = s_next, m_next
                transition += m
            metrics['episode_reward'].append(episode_reward)
            metrics['num_acquisition'].append(num_acquisition)
            transitions.append(transition.astype(np.int32))
            # evaluate the final state
            p = self.model.predict(s, m)
            x = self.env.x
            x = np.expand_dims(np.absolute(np.fft.ifft2(np.fft.ifftshift(x[...,0] + x[...,1] * 1j, axes=(-2,-1)))), axis=-1)
            mse = np.sum(np.square(p-x), axis=tuple(range(1,p.ndim)))
            metrics['mse'].append(mse)
            ssim = compute_ssim(p, x)
            metrics['ssim'].append(ssim)
            # zero filling
            pz = s * m
            pz = np.expand_dims(np.absolute(np.fft.ifft2(np.fft.ifftshift(pz[...,0] + pz[...,1] * 1j, axes=(-2,-1)))), axis=-1)
            mse = np.sum(np.square(pz-x), axis=tuple(range(1,p.ndim)))
            metrics['mse_zero'].append(mse)
            ssim = compute_ssim(pz, x)
            metrics['ssim_zero'].append(ssim)
            # plot results
            save_dir = f'{self.args.exp_dir}/evaluate/{folder}'
            os.makedirs(save_dir, exist_ok=True)
            gt = x[0,:,:,0]
            pred = p[0,:,:,0]
            pred_zero = pz[0,:,:,0]
            mask = m[0,:,:,0]
            img = np.concatenate([mask, pred, pred_zero, gt], axis=1)
            plt.imsave(f'{save_dir}/{num_batches}.png', img)

        # concat metrics
        average_metrics = defaultdict(float)
        for k, v in metrics.items():
            metrics[k] = np.concatenate(v)
            average_metrics[k] = np.mean(metrics[k])

        # log
        logger.info('#'*20)
        logger.info('evaluate:')
        for k, v in average_metrics.items():
            logger.info(f'{k}: {v}')

        return {'metrics': metrics, 'transitions': transitions}


