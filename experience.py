"""Experience buffer for experience replay."""

from collections import deque
import random
import numpy as np

class Experience:
    """Experience buffer to store and sample experience.

    Attributes:
        batch_size (int): Size of batch to sample.
        buffer (deque): List to store experience.
    """

    def __init__(self, size, batch_size, rand_seed=1234):
        """Initialize a replay buffer.

        Args:
            size (int): Maximum size of buffer.
            batch_size (int): Batch size.
            rand_seed (int): Seed for sampler.
        """
        self.batch_size = batch_size
        self.buffer = deque(maxlen=size)
        random.seed(rand_seed)

    def add(self, x, a, r, d, x_next):
        """Add experience of (x,a,r,d,x_next) format to buffer.

        Args:
            x (ndarray): State array.
            a (ndarray): Actions array.
            r (float): Reward.
            d (bool): Terminal flag.
            x_next (ndarray): Next state array.

        Returns:
            void
        """
        experience = (x, a, np.array([r]), np.array([d]), x_next)
        self.buffer.append(experience)

    def sample_batch(self):
        """Sample a batch of size batch_size.

        Args:
            void

        Returns:
            - (list): Experience list containing sampled experience, same format as stored.
        """
        if len(self.buffer) < self.batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, self.batch_size)

        x_batch = np.array([sample[0] for sample in batch])
        a_batch = np.array([sample[1] for sample in batch])
        r_batch = np.array([sample[2] for sample in batch])
        d_batch = np.array([sample[3] for sample in batch])
        x_next_batch = np.array([sample[4] for sample in batch])

        return x_batch, a_batch, r_batch, d_batch, x_next_batch

    def clear_buffer(self):
        """Clear the buffer.

        Args:
            void

        Return:
            void
        """
        self.buffer.clear()
