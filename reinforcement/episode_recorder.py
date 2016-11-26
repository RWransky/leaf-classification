import numpy as np
import random


class episode_recorder():
    def __init__(self, buffer_size=50):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            # self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_episodes = extract_valid_episode(self.buffer, batch_size, self.buffer_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, abs(len(episode)-trace_length))
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size*trace_length, 5])


def extract_valid_episode(ep_buffer, batch_size, buffer_size):
    valid = False
    while valid is False:
        valid = True
        valid_episodes = []
        sample_indx = random.sample(range(buffer_size), batch_size)
        for indx in sample_indx:
            if len(ep_buffer[indx]) > 0:
                valid_episodes.append(ep_buffer[indx])
            else:
                valid = False
                break
    return valid_episodes
