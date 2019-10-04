import random

import torch


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """Сохраняет элемент в циклический буфер"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Возвращает случайную выборку указанного размера"""

        def get_tensor(data):
            need_tensor = torch.tensor(data, dtype=torch.float)
            if len(need_tensor.size()) == 1:
                return need_tensor.reshape(batch_size, -1)  # , -1)
            else:
                return need_tensor  # .transpose_(0, 1)

        return list(
            map(lambda elem: get_tensor(elem), zip(*random.sample(self.memory, batch_size))))

    def __len__(self):
        return len(self.memory)
