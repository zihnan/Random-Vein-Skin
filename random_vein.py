import math
import numpy as np

def relu(matrix):
     return np.maximum(matrix, 0)

def logical_bool(matrix):
    return np.greater(matrix, 0)

def expand(matrix, margin=0, top_margin=0, bottom_margin=0, left_margin=0, right_margin=0):
    expand_matrix = np.zeros((matrix.shape[0] + margin*2 + (top_margin + bottom_margin), matrix.shape[1] + margin*2 + (left_margin + right_margin)))
    print(expand_matrix.shape)
    expand_matrix[margin + top_margin:matrix.shape[0] + margin + top_margin, margin + left_margin:matrix.shape[1] + margin + left_margin] = matrix
    return expand_matrix

def conv_2d(expanded_matrix, shape, top_margin=1, right_margin=1, bottom_margin=1, left_margin=1, mask=np.array([[1, 1,1], [1,1, 1], [1,1,1]])):
    conv_matrix = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            position_y = top_margin + i
            position_x = left_margin + j
            temp = expanded_matrix[position_y - math.ceil(mask.shape[0]/2)+1:position_y + math.ceil(mask.shape[0]/2), position_x - math.ceil(mask.shape[1]/2)+1:position_x + math.ceil(mask.shape[1]/2)].reshape(mask.shape)
            print(position_y, position_x)
            conv_matrix[i, j] = np.sum(temp * mask)
    return conv_matrix

def random_vein_lines():
    offset = 1
    skin = np.random.randn(4,12)
    relu_skin = np.maximum(skin, 0)
    expand_relu_skin = expand(relu_skin, 1)
    cross1 = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]])
    cross2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    cross1 = np.reshape(cross1, (-1,))
    cross2 = np.reshape(cross2, (-1,))
    result = np.zeros(skin.shape)
    for i in range(skin.shape[0]):
         for j in range(skin.shape[1]):
             position_y = offset + j
             position_x = offset + i
             a = expand_relu_skin[position_x - 1:position_x +2,position_y - 1: position_y + 2]
             a = np.reshape(a, (-1,))
             print(a)
             print(position_x - 1,position_x + 2)
             print(position_y - 1, position_y + 2)
             print(np.dot(a, cross1.T))
             print(np.dot(a, cross2.T))
             temp1 = relu(np.dot(a, cross1.T))
             temp2 = relu(np.dot(a, cross2.T))
             print(temp1 , temp2)
             print(np.logical_xor(temp1 > 0,temp2 > 0))
             result[i, j] = np.logical_xor(temp1 > 0,temp2 > 0)
             print('>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(relu(skin))
    print(logical_bool(skin))
    return result, skin