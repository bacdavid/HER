"""Hindsight Experience Replay implementation with future replay."""

import time
import numpy as np
from collections import deque
from utils import concat

class HER:
    """HER agent implementation https://arxiv.org/abs/1707.01495.

    Attributes:
        sess (): Tensorflow session.
        buffer (): Buffer for experience replay.
        env (): Environment for the agent.
        actor (): Actor that maps [s, g] to a.
        critic (): Critic that maps [s, g], a to a state-action value.
        actor_noise (): Noise for state exploration.
        r_mon (): Monitor instance for recording values in tensorboard.
        ep_buffer (deque): List to store episode experience
      """

    def __init__(self, saver, buffer, env, actor, critic, actor_noise):
        """Initialize HER agent.

        Args:
            buffer(): Buffer.
            env (): Environment.
            actor (): Actor.
            citic (): Critic.
            actor_noise (): Exploration noise.
            r_mon (): Reward monitor.
        """
        self.saver = saver
        self.buffer = buffer
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.ep_buffer = deque()

    def train(self, gamma, her_k, max_episodes, max_episode_len, replay_len):
        """Train HER agent

        Args:
            gamma (float): Discount factor for future reward.
            her_k (int): Number of HER replays per experience sample.
            max_episodes (int): Max number of episodes to train.
            max_episode_len (int): Max number of steps in ep.
            replay_len (int): Number of mini batches to replay.

        Returns:
            void
        """

        # initialize target networks
        self.actor.copy_vars()
        self.critic.copy_vars()
        total_r = 0

        for i in range(max_episodes):
            # reset if terminated
            s = self.env.reset()
            g = self.env.sample_goal()
            self.actor_noise.reset()
            self.ep_buffer.clear()

            # concat
            s_concat = concat(s, g)

            # write to monitor
            print('episode ' + str(i) + ' reward ' + str(total_r))
            total_r = 0
            
            for j in range(max_episode_len):

                # predict action
                a = self.actor_noise.predict_action(s_concat)

                # take action
                s_next, r, d, _ = self.env.step(a)
                s_next_concat = concat(s_next, g)
                total_r += r

                # add sample to episode buffer
                self.ep_buffer.append((s, a, r, d, s_next))
                self.buffer.add(s_concat, a, r, d, s_next_concat)

                s = s_next
                s_concat = s_next_concat
                if d:
                    break

            # HER
            T = len(self.ep_buffer)
            for t, (s, a, _, _, s_next) in enumerate(self.ep_buffer):
                for _ in range(her_k):
                    future = np.random.randint(t, T)
                    _, _, _, _, s_future = self.ep_buffer[future]
                    g = s_future
                    r, d = self.env.reward(g, s)
                    s_concat = concat(s, g)
                    s_next_concat = concat(s_next, g)
                    self.buffer.add(s_concat, a, r, d, s_next_concat)

            # Optimize
            for _ in range(replay_len):

                # sample from buffer
                s_batch, a_batch, r_batch, d_batch, s_next_batch = self.buffer.sample_batch()

                # predict all q_next
                q_next_batch = self.critic.predict(s_next_batch, self.actor.predict_target(s_next_batch))

                # create a mask wrt terminal states
                mask = (np.logical_not(d_batch)).astype(np.float32)

                # create target values
                y_batch = r_batch + gamma * q_next_batch * mask

                # train the critic
                self.critic.train(s_batch, a_batch, q_target=y_batch)

                # get new batch of actions
                new_actions = self.actor.predict(s_batch)

                # update the actor
                actions_grads = self.critic.get_gradients(s_batch, new_actions)
                self.actor.train(s_batch, actions_grads)

                # update target networks
                self.actor.update_vars()
                self.critic.update_vars()

            self.saver.save_model(name_of_event=str(i))

    def play(self, max_episodes, max_episode_len):

        total_r = 0

        for i in range(max_episodes):

            # reset if terminated
            s = self.env.reset()
            g = self.env.sample_goal()
            self.env.render_goal()
            time.sleep(2)
            self.env.close_goal()

            # concat
            s_concat = concat(s, g)

            # write to monitor
            print('episode ' + str(i) + ' reward ' + str(total_r))
            total_r = 0

            for j in range(max_episode_len):

                # predict action
                a = self.actor.predict_target([s_concat])[0]

                # take action
                s_next, r, d, _ = self.env.step(a)
                self.env.render()
                s_next_concat = concat(s_next, g)
                total_r += r

                s_concat = s_next_concat
                if d:
                    break

            self.env.close()

    def _g_map(self, g):
        pass