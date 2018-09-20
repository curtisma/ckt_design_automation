import numpy as np
import tensorflow as tf
import random

class BatchGenerator(object):
    def __init__(self, data_set_size, batch_size):
        self._data_size = data_set_size
        self._batch_size = batch_size
        self._segment = self._data_size // batch_size
        self.last_index = 0
        self._permutations = list(range(data_set_size))
        random.shuffle(self._permutations)

    def next(self):

            if ((self.last_index+1)*self._batch_size > self._data_size):
                indices1 = self._permutations[self.last_index * self._batch_size:]
                indices2 = self._permutations[:((self.last_index+1)*self._batch_size)%self._data_size]
                indices = indices1 + indices2
            else:
                indices = self._permutations[self.last_index * self._batch_size:(self.last_index + 1) * self._batch_size]

            self.last_index = (self.last_index+1) % (self._segment+1)
            return indices


class Design(list):
    def __init__(self, seq=()):
        list.__init__(self, seq)
        self.cost =     None
        self.fitness =  None
        self.specs =    dict(
            ugbw_cur=   None,
            gain_cur=   None,
            phm_cur=    None,
            tset_cur=   None,
            psrr_cur=   None,
            cmrr_cur=   None,
            offset_curr=None,
            ibias_cur=  None,
        )

    @property
    def cost(self):
        return self.__cost

    @property
    def fitness(self):
        return self.__fitness

    @cost.setter
    def cost(self, x):
        self.__cost = x
        self.__fitness = -x if x is not None else None

    @fitness.setter
    def fitness(self, x):
        self.__fitness = x
        self.__cost = -x if x is not None else None