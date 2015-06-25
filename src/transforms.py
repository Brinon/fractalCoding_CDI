import numpy as np
'''
Range blocks -> non overlaping, NxN
Domain blocks -> may be overlaping, N'x N', N' > N
'''
def affine_transform(p, transform_matrix):
    '''
    :param p_ini: point to be transformed, x,y position, z pixel intensity
    :param transform_matrix: matrix with shape 3,3,
    [[k11,k12,0],
     [k21,k22,0],
     [0, 0, a]]
    '''
    return np.dot(transform_matrix, p)

def reconstruct_img(blocks):
    ''' blocks is a N*N square matrix '''
    res = []
    for row in blocks:
        # row array de blocks quadrats 
        for fila in range(len(row[0])):
            r = []
            for b in range(len(row)):
                r += row[b][fila] 
            res.append(r)
    return np.vstack(res)

def main():
    pass

