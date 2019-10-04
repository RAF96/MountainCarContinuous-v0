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
        return list(
            map(lambda elem: torch.Tensor(elem).reshape(batch_size, -1), zip(*random.sample(self.memory, batch_size))))

    def __len__(self):
        return len(self.memory)
