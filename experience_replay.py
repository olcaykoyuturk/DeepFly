# Deneyim Tekrarı için Belleği Tanımla
from collections import deque
import random
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # İsteğe bağlı olarak tekrarlanabilirlik için seed değeri
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)