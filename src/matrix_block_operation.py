from PIL import Image
import numpy as np
import os,sys
import math
from transforms import *
image_path = '../samples/lena.bmp'

def decomp_blocks(image, block_size):
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
    return domain_blocks


def get_range_block(image, p, w=8, h=8):
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


def rotate(mat, times):
    return np.rot90(mat,-times)

# augmentar/disminuir la brillantor segons un factor
def brightness(mat, fact):
    img = np.zeros((len(mat), len(mat[0])))
    # img = [[0]*len(mat[0]) for _ in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            val = mat[i][j]*fact
            if (val > 255):
                img[i][j] = 255.0
            else:
                img[i][j] = val
    return np.asarray(img)

# passa a imatge
def asImage(mat):
    return Image.fromarray(mat)

def similarity(mat1, mat2):
    count = 0
    # mats tenen mateixa mida
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            if mat1[i][j] == mat2[i][j]:
                count += 1
    return count/(len(mat1)*len(mat1[0]))

# redueix a la meitat una imatge fent la mitjana de color de cada 2 pixels (la mida ha de ser parell)
def reduce_matrix(mat, n=2):
    new_mat = np.zeros((len(mat)//n, len(mat[0])//n))
    for i in range(len(mat)//n):
        for j in range(len(mat[0])//n):
            new_mat[i][j] = round((mat[2*i][2*j]+mat[2*i+1][2*j+1])/n)
    return np.asarray(new_mat)

def main():
    im = Image.open(image_path)
    # print (im.size)
    b = decomp_domain_blocks(im,16)
    # print(b)
    # print(len(b), len(b[0]))
    # print(np.asarray(im)[0][0:8], np.asarray(im)[1][0:8])
    # print(b[0][0])
    rb = np.zeros((len(b),len(b[0]),len(b[0][0])//2,len(b[0][0][0])//2))
    b1 = get_range_block(im, (0,0), 2,4)
    for i in range(len(b)):
        for j in range(len(b[0])):
            rb[i][j] = reduce_matrix(np.asarray(b[i][j],dtype='float32'))

    # asImage(reduce_domain(np.asarray(im))).show()
    # print(b1)



def check_transforms(rangeb, domainb, threshold):
   
    if similarity(rangeb, rotate90(domainb)) <= threshold:
        return 'r90'
    elif similarity(rangeb, rotate180(domainb)) <= threshold:
        return 'r180'
    elif similarity(rangeb, rotate270(domainb)) <= threshold:
        return 'r270'
    elif similarity(rangeb, flip_vertical(domainb)) <= threshold:
        return 'fvr'
    elif similarity(rangeb, flip_horizontal(domainb)) <= threshold:
        return 'fhr'
    return -1

def undo_trans(block, trans):
    if trans == 'r90':
        return rotate90(block)
    elif trans == 'r180':
        return rotate180(block)
    elif trans == 'r270':
        return rotate270(block)
    elif trans == 'fvr':
        return flip_vertical(block)
    elif trans == 'fhr':   
        return flip_horizontal(block)
  
 
def main2():
    im = Image.open(image_path)

    domain_blocks = decomp_blocks(im, 16)
    range_blocks = decomp_blocks(im, 8)
    print(len(range_blocks), len(range_blocks[0]), len(range_blocks[0][0]), len(range_blocks[0][0][0]))
    im = asImage(reconstruct_img(range_blocks))
    im.show()
    range_mapping = {}
    result_img = np.array([]) 
    #for each range
    for i in range(len(range_blocks)):
        fil = np.array([])
        for j in range(len(range_blocks[0])):
            range_act = range_blocks[i][j]
            ### buscar un domain block
            block_found = False

            for d_i in range(len(domain_blocks)):
                for d_j in range(len(domain_blocks[0])):
                    while not block_found:
                        domain_act = reduce_matrix(domain_blocks[d_i][d_j])
                        res = check_transforms(range_act, domain_act, 0.1)
                        if res != -1:
                            range_mapping[(i,j)] = res
                 #           print ('block {},{} transformed: ({},{},{})'.format(i,j,d_i,d_j, res))
                            fil=np.append(fil, undo_trans(domain_act, res))
                            block_found = True
        result_img = np.append(result_img, fil)
    result_img = result_img.reshape((64,64,8,8))
    print (result_img)
    print(type(result_img), result_img.shape)
    res_img = asImage(reconstruct_img(result_img))
    res_img.show()
    
if __name__ == '__main__':
    main2()
