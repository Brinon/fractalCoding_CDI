from PIL import Image
import numpy as np
import os,sys
import math
from transforms import *

image_path = 'samples/lena.jpg'

def get_range_blocks(image, block_size=8):
    '''
    :param image: PIL Image
    divide the image into blocks height x width 
    '''
    w,h = image.size
    im_arr = np.asarray(image) # array of shape 512,512
    n_blocks_v = w // block_size
    n_blocks_h = w // block_size
    domain_blocks = []  #matrix that will store the domain blocks
    for i in range(n_blocks_v):
        domain_row = []
        for j in range(n_blocks_h):
            # block i,j
            block_ij = []
            for ii in range(block_size):
                fila_i = []
                for jj in range(block_size):
                     fila_i += [im_arr[(i * block_size) + ii][(j * block_size) + jj]]
                block_ij += [fila_i]
            domain_row += [block_ij]
        domain_blocks += [domain_row]
    return domain_blocks.reshape((64,64))


def get_domain_block(image, p, w=16, h=16):
    '''
    :param p: tuple x,y 
    return the block with left top corner at p of dimensions WxH '''
    x,y = p
    b = []
    im_arr = np.array(image)
    for i in range(w):
        fila = []
        for j in range(h):
            fila += [im_arr[x + i][y + j]]
        b += [fila]
    return b

def get_domain_blocks(image,w=16,h=16):
    im_arr = np.array(image)
    dom_blocks = np.array([])
    # for i in range(len(im_arr)-h):
    #     for j in range(len(im_arr[0])-w):
    for i in range(0,len(im_arr),h):
        for j in range(0,len(im_arr[0]),w):
            dom_blocks = np.append(dom_blocks,get_domain_block(image,(i,j),w,h))
    return dom_blocks


# passa a imatge
def asImage(mat):
    return Image.fromarray(mat)

# redueix a la meitat una imatge fent la mitjana de color de cada 2 pixels (la mida ha de ser parell)
def reduce_matrix(mat, n=2):
    new_mat = np.zeros((len(mat)//n, len(mat[0])//n))
    for i in range(len(mat)//n):
        for j in range(len(mat[0])//n):
            new_mat[i][j] = round((mat[2*i][2*j]+mat[2*i+1][2*j+1])/n)
    return np.asarray(new_mat)

def transform(mat):
    return np.asarray([brightness(mat, 0.1),brightness(mat, -0.1)])

def main():
    im = Image.open(image_path)
    # print (im.size)
    b1 = get_domain_blocks(im)
    # print(b)
    # print(len(b), len(b[0]))
    # print(np.asarray(im)[0][0:8], np.asarray(im)[1][0:8])
    # print(b[0][0])
    b = get_range_blocks(im)
    red_dom = np.zeros((len(b),len(b[0]),len(b[0][0])//2,len(b[0][0][0])//2))
    for i in range(len(b)):
        for j in range(len(b[0])):
            red_dom[i][j] = reduce_matrix(np.asarray(b[i][j],dtype='float32'))

    pos = []
    trans = []
    final = np.array([])
    for k1 in range(len(b1)):
        for k2 in range(len(b1[0]))
            lvl = 0
            print(k)
            for i in range(len(red_dom)):
                for j in range(len(red_dom[0])):
                    transforms = transform(red_dom[i][j])
                    for t in range(len(transforms)):
                        lvl2 = similarity(b1[k],t)
                        if lvl2 < lvl:
                            lvl = lvl2
                            pos[k] = i,j
                            trans[k] = t
                            final = np.append(k,transforms[t])
    print("Pos:",pos)
    print("Trans:",trans)
    asImage(np.asarray(reconstruct_img(final.reshape((32,32))))).show()
    # asImage(reduce_domain(np.asarray(im))).show()
    # print(b1)

if __name__ == '__main__':
    main()
