
import numpy as np
import pandas as pd
import tensorflow as tf

class DeepQ
    def __init__(self):
        pass

    def choose_action(self):
        if np.random.rand( ) <=epsilon:  # random
            action = np.random.randint(0 ,n_action)
        else  :# learned action
            # Q = model.predict(obs)
            Q = sess.run(out, {tfx:obs})
            action = np.argmax(Q)

