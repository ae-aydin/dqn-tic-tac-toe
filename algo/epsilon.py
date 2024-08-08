from abc import ABC, abstractmethod


# Base Class for epsilon (exploration / exploitation) value operations.
class BaseEpsilon(ABC):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        return self.epsilon


# Class for keeping epsilon constant.
class ConstantEpsilon(BaseEpsilon):
    def __init__(self, epsilon: float):
        super().__init__(epsilon)

    def step(self):
        return self.epsilon

    @property
    def value(self):
        return self.epsilon


# Class for decaying epsilon linearly, tracking steps taken.
class EpsilonLinearDecay(BaseEpsilon):
    def __init__(self, epsilon: float, final_epsilon: float, total_steps):
        super().__init__(epsilon)
        self.final_epsilon = final_epsilon
        self.total_steps = total_steps
        self.steps_taken = 0
        self.decay = (final_epsilon - epsilon) / total_steps

    def step(self):
        self.steps_taken += 1
        self.epsilon = (
            self.epsilon + self.decay
            if self.steps_taken < self.total_steps
            else self.final_epsilon
        )
        return self.epsilon

    @property
    def value(self):
        return self.epsilon


# TODO
# Class for decaying epsilon exponentially.
class EpsilonExpDecay(BaseEpsilon):
    def __init__(self, epsilon: float):
        super().__init__(epsilon)
