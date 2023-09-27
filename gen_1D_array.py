import numpy as np
import os

import pandas as pd


#
n = 10000

class Gen1darray:

    def __init__(self):
        pass


    def _gen_1d_array(self,num_arrays, array_size):
        for i in range(1, num_arrays + 1):
            random_array = np.random.randint(1, 9, size=array_size)

            file_name = f"arrays/map200_{i}.txt"

            np.savetxt(file_name, random_array, fmt="%d", delimiter="\n", encoding='UTF-8')

        print("Arrays saved successfully.")

Gen1darray()._gen_1d_array(n, 32)


# data = pd.read_csv('C:/Users/Administrator/Desktop/CNN1d/merged_files.txt', header=None)
#
# reshape_data = data.values.reshape(10000, 32)
# reshape_data = pd.DataFrame(reshape_data)
# reshape_data.to_csv('C:/Users/Administrator/Desktop/CNN1d/all_str.csv', header=None, index=None)

# with open('C:/Users/Administrator/Desktop/CNN1d/all_str.txt', 'w') as f:
#     f.write(reshape_data)
#     f.close()

# data1 = pd.read_csv('C:/Users/Administrator/Desktop/CNN1d/all_str.csv', header=None)
#
# data2 = pd.read_csv('C:/Users/Administrator/Desktop/CNN1d/data/new_test.csv', header=None)
# # data2_header = pd.read_csv('C:/Users/Administrator/Desktop/CNN1d/data/new_test.csv', header=None, nrows=0)
#
#
# data2 = data2.drop(data2.columns[0],axis=1)
#
# print(data2.shape)
# data2 = data2.drop(data2.index[0],axis=0)
#
#
#
# data_total = pd.concat([data2, data1], axis=1)
# print(data_total.shape, data_total.head(5))
# data_total.to_csv('C:/Users/Administrator/Desktop/CNN1d/data/total.csv', header=None, index=None)
