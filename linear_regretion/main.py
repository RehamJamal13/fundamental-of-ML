import numpy as np
from dataset import generate_data
from model import Linearregretion


Xtrain, ytrain = generate_data()
model = Linearregretion()

model.train_batch_gradient_descent(Xtrain, ytrain)