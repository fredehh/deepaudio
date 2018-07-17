
import numpy as np

def batchloader(name, annotation_list, data_location, x_temp1, x_temp2):
    temp = annotation_list[:,-1].reshape((len(annotation_list[:,-1]),1))
    temp = (temp+1)/2
    y = np.zeros((len(annotation_list),2))
    for i in range(len(annotation_list)):
        x_temp1[i] = np.load(data_location+str(annotation_list[i][0]).zfill(4)+name+'.npy')
        x_temp2[i] = np.load(data_location+str(annotation_list[i][1]).zfill(4)+name+'.npy')
        if temp[i] == 0:
            y[i,0] = 1
        else:
            y[i,1] = 1
    return x_temp1, x_temp2, y
