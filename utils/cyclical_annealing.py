import math

import matplotlib.pyplot as plt
import numpy as np


class CyclicalScheduler:
    def __init__(self, type: str, start: float, stop: float, n_step: int, cycle: int = 4, ratio: float = 0.5):
        self.type = type
        self.start = start
        self.stop = stop
        self.n_step = n_step
        self.cycle = cycle
        self.ratio = ratio
        self.temps = np.ones(n_step) * stop
        self.period = n_step / cycle
        self.diff = stop - start
        self.step_value = self.diff / (self.period * ratio)
        self.sig_mult = 6.0 / (self.diff * 0.5)

        self.update_functions = {
            "linear": lambda v: v,
            "sigmoid": lambda v: (1.0 / (1.0 + np.exp(-(v - self.start - self.diff * 0.5) * self.sig_mult))) * self.diff + self.start,
            "cosine": lambda v: 0.5 * (1.0 - math.cos(math.pi * (v - self.start) / self.diff)) * self.diff + self.start,
        }

        self.init_temps()

    def init_temps(self):
        for c in range(self.cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * self.period) < self.n_step):
                self.temps[int(i + c * self.period)] = self.update_functions[self.type](v)
                v += self.step_value
                i += 1
        return

    def get_temps(self):
        return self.temps

    def get_temp(self, step: int):
        return self.temps[step]

    def __call__(self, step: int):
        return self.get_temp(step)

    def __len__(self):
        return self.n_step

    def plot(self):
        fig = plt.figure(figsize=(8, 4.0))
        stride = max(int(self.n_step / 8), 1)
        plt.plot(range(self.n_step), self.temps, "-", label=self.type, marker="s", color="k", markevery=stride, lw=2, mec="k", mew=1, markersize=10)
        plt.grid(True)
        plt.show()
        return


if __name__ == "__main__":
    type = "linear"
    CyclicalScheduler(type=type, start=0, stop=1, n_step=4000, cycle=4, ratio=0.5).plot()
    CyclicalScheduler(type=type, start=0.1, stop=0.5, n_step=4000, cycle=1, ratio=0.25).plot()

    type = "sigmoid"
    CyclicalScheduler(type=type, start=8, stop=10, n_step=4000, cycle=4, ratio=1).plot()
    CyclicalScheduler(type=type, start=1, stop=10, n_step=400, cycle=4, ratio=0.5).plot()
    CyclicalScheduler(type=type, start=1, stop=10, n_step=400, cycle=2, ratio=0.5).plot()

    type = "cosine"
    CyclicalScheduler(type=type, start=0, stop=1, n_step=4000, cycle=4, ratio=0.5).plot()
    CyclicalScheduler(type=type, start=10, stop=30, n_step=4000, cycle=1, ratio=0.25).plot()
