#%%
import numpy as np
import pandas as pd
import pickle


CSV_1 = 'G:/subject2/accel/data3/train.csv'
CSV_2 = 'G:/subject2/accel/data3/test.csv'

TRAIN_SET = 'G:/subject2/accel/class1/train_set.pickle'
TEST_SET = 'G:/subject2/accel/class1/test_set.pickle'

raw_1 = pd.read_csv(CSV_1, header=None)
raw_2 = pd.read_csv(CSV_2, header=None)




raw_1 = raw_1.values
raw_2 = raw_2.values


x_bal = raw_1[:, :-1]
y_bal = raw_1[:,-1].astype(int)
del raw_1
x_test = raw_2[:, :-1]
y_test = raw_2[:,-1].astype(int)
del raw_2
print('Training set shapes:', np.shape(x_bal), np.shape(y_bal))
print('Test set shapes:', np.shape(x_test), np.shape(y_test))

#%%


with open(TEST_SET, 'wb') as file:
    pickle.dump({'x': x_test, 'y': y_test}, file)

with open(TRAIN_SET, 'wb') as file:
    pickle.dump({'x': x_bal, 'y': y_bal}, file)
# %%
