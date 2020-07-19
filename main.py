"""Train a HER agent to play the tiling puzzle."""

import argparse
import tensorflow as tf
import gym
import numpy as np
from hindsight_experience import HER
from monitor import Monitor
from model import Model
from actor_critic import Actor, Critic
from experience import Experience
from noise import ActorNoise
from test_env import TestEnv

def main(args):
    with tf.device(args['device']):

        # tf
        tf.set_random_seed(args['rand_seed'])
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # env
        env = gym.make('TestEnv-v0')
        env.seed(args['rand_seed'])
        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        concat_dim = 2
        batched_s_dim = [None, s_dim, concat_dim]
        batched_a_dim = [None, a_dim]

        # agents
        actor = Actor(sess, args['actor_lr'], args['tau'], args['batch_size'], args['clip_val'], batched_s_dim,
                      batched_a_dim)
        critic = Critic(sess, args['critic_lr'], args['tau'], args['clip_val'], batched_s_dim, batched_a_dim)

        # experience
        exp = Experience(args['buffer_size'], args['batch_size'], args['rand_seed'])

        # noise
        actor_noise = ActorNoise(actor.predict, a_dim,  noise_type=args['noise_type'])

        # initialize
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = Model(sess, args['restore_path'])
        saver.restore_model()

        # training
        her = HER(saver, exp, env, actor, critic, actor_noise)
        if args['mode'] == 'train':
            her.train(args['gamma'], args['her_k'], args['max_episodes'], args['max_episode_len'], args['replay_len'])
        else:
            her.play(args['max_episodes'], args['max_episode_len'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='her')

    # agent parameters
    parser.add_argument('--actor_lr', help='actor network learning rate', default=0.001, type=float)
    parser.add_argument('--critic_lr', help='critic network learning rate', default=0.001, type=float)
    parser.add_argument('--gamma', help='discount factor', default=0.99, type=float)
    parser.add_argument('--tau', help='inertia factor of target network', default=0.001, type=float)
    parser.add_argument('--buffer_size', help='maximum buffer size', default=1000000, type=int)
    parser.add_argument('--batch_size', help='batch size for training', default=256, type=int)
    parser.add_argument('--clip_val', help='gradient clip', default=40.0, type=float)
    parser.add_argument('--noise_type', help='OU or epsg', default='epsg')

    # training parameters
    parser.add_argument('--device', help='device for training', default='/gpu:0')
    parser.add_argument('--rand_seed', help='random seed', default=1234, type=int)
    parser.add_argument('--her_k', help='k for HER replay', default=8, type=int)
    parser.add_argument('--max_episodes', help='number of total episodes', default=500, type=int)
    parser.add_argument('--max_episode_len', help='max episode length', default=1000, type=int)
    parser.add_argument('--replay_len', help='replay frequency', default=40, type=int)

    # set up
    parser.add_argument('--restore_path', help='path for model to restore', default='./restore')
    parser.add_argument('--mode', help='play or train', default='train')


    args = vars(parser.parse_args())
    main(args)
