import math
import numpy as np

'''

_, skin = random_vein_lines()
tf = expand(skin, top_margin=1, left_margin=1)
tr = expand(skin, top_margin=1, right_margin=1)
bf = expand(skin, bottom_margin=1, left_margin=1)
br = expand(skin, right_margin=1, bottom_margin=1)
ex = expand(skin, margin=1)

mask = np.array([[1, 1], [1, 1]])
tfc = conv_2d(tf, skin.shape, top_margin=1, left_margin=1, mask=mask)
trc = conv_2d(tr, skin.shape, top_margin=1, right_margin=1, mask=mask)
bfc = conv_2d(bf, skin.shape, bottom_margin=1, left_margin=1, mask=mask)
brc = conv_2d(br, skin.shape, right_margin=1, bottom_margin=1, mask=mask)
exc = conv_2d(ex, skin.shape)

'''

def relu(matrix):
     return np.maximum(matrix, 0)

def logical_bool(matrix):
    return np.greater(matrix, 0)

def expand(matrix, margin=0, top_margin=0, bottom_margin=0, left_margin=0, right_margin=0):
    expand_matrix = np.zeros((matrix.shape[0] + margin*2 + (top_margin + bottom_margin), matrix.shape[1] + margin*2 + (left_margin + right_margin)))
    expand_matrix[margin + top_margin:matrix.shape[0] + margin + top_margin, margin + left_margin:matrix.shape[1] + margin + left_margin] = matrix
    return expand_matrix

def conv_2d(expanded_matrix, shape, top_margin=1, right_margin=1, bottom_margin=1, left_margin=1, mask=np.array([[1, 1,1], [1,1, 1], [1,1,1]])):
    conv_matrix = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            position_y = top_margin + i
            position_x = left_margin + j
            temp = expanded_matrix[position_y - int(math.floor(mask.shape[0]/2.0)):position_y + int(math.ceil(mask.shape[0]/2.0)), position_x - int(math.floor(mask.shape[1]/2.0)):position_x + int(math.ceil(mask.shape[1]/2.0))].reshape(mask.shape)
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
            temp1 = relu(np.dot(a, cross1.T))
            temp2 = relu(np.dot(a, cross2.T))
            result[i, j] = np.logical_xor(temp1 > 0,temp2 > 0)
    return result, skin
    
def random_vein_lines2():
    skin = np.random.randn(4,12)
    expanded_skin = expand(skin, margin=1)
    mask1 = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]])
    mask2 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    result = np.zeros(skin.shape)
    ex1 = conv_2d(expanded_skin, skin.shape, mask=mask1)
    ex1 = relu(ex1)
    ex2 = conv_2d(expanded_skin, skin.shape, mask=mask2)
    ex2 = relu(ex2)
    result = np.logical_xor(ex1, ex2)
    return result, skin
    
def random_vein_lines3():
    skin = np.random.randn(4,12)
    expanded_skin = expand(skin, margin=1)
    mask = np.array([[1, 1],
                        [1, 1]])
    result = np.zeros(skin.shape)
    tf = expand(skin, top_margin=1, left_margin=1)
    tr = expand(skin, top_margin=1, right_margin=1)
    bf = expand(skin, bottom_margin=1, left_margin=1)
    br = expand(skin, right_margin=1, bottom_margin=1)
    
    tfc = conv_2d(tf, skin.shape, top_margin=1, left_margin=1, mask=mask)
    trc = conv_2d(tr, skin.shape, top_margin=1, right_margin=1, mask=mask)
    bfc = conv_2d(bf, skin.shape, bottom_margin=1, left_margin=1, mask=mask)
    brc = conv_2d(br, skin.shape, right_margin=1, bottom_margin=1, mask=mask)
    
    tfc = relu(tfc)
    trc = relu(trc)
    bfc = relu(bfc)
    brc = relu(brc)
    
    temp1 = np.logical_xor(tfc, trc)
    temp2 = np.logical_xor(bfc, brc)
    result = np.logical_xor(temp1, temp2)
    return result, skin
