from PIL import Image
import numpy as np
import time, os,sys
import math
import pickle
from transforms import *
from optparse import OptionParser

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

def decomp_overlaping_blocks(image, block_size):
    w,h = image.size
    im_arr = np.asarray(image)
    blocks = []
    n_blocks_v = w - block_size + 1
    n_blocks_h = h - block_size + 1
    for i in range(n_blocks_v):
        row = []
        for j in range(n_blocks_h):
            block_ij = get_block_at(im_arr, (i,j), block_size, block_size)
            row += [block_ij]
        blocks += [row]
    return blocks


def get_block_at(im_arr, p, w=8, h=8):
    '''
    :param p: tuple x,y
    return the block with left top corner at p of dimensions WxH '''
    x,y = p
    b = []
    #im_arr = np.array(image)
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

# redueix a la meitat una imatge fent la mitjana de color de cada 2 pixels (la mida ha de ser parell)
def reduce_matrix(mat, n=2):
    #print('mat shape: ', mat.shape )
    new_mat = np.zeros((len(mat)//n, len(mat[0])//n))
    for i in range(len(mat)//n):
        for j in range(len(mat[0])//n):
            new_mat[i][j] = round((mat[2*i][2*j]+mat[2*i+1][2*j+1])/n)

    return np.asarray(new_mat)

def inverse(block, trans):
    if trans == 'r90':
        return np.rot90(block,1)
    elif trans == 'r180':
        return np.rot90(block,2)
    elif trans == 'r270':
        return np.rot90(block,-1)
    elif trans == 'fvr':
        return flip_horizontal(block)
    elif trans == 'fhr':
        return flip_vertical(block)
    elif trans == 'b15':
        return brightness(block,-0.15)
    elif trans == 'b20':
        return brightness(block,-0.2)
    else:
        return block

def decode(blocks, transforms):
    decoded_img = np.asarray([])
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            decoded_img = np.append(decoded_img, inverse(blocks[i][j], transforms[(i,j)]))
    # print(decoded_img.shape)
    return reconstruct_img(decoded_img.reshape((64,64,8,8)))
    # reconstruct_img(decoded_img)

def encode_image(image_path, result_path, n_its=1):
    im = Image.open(image_path)
    im = im.convert('L')
    tini = time.clock()
    res_img, range_map = fractal(im, result_path, 30,False)
    tend = time.clock()
    print('Size of compressed image: {}'.format(
        sys.getsizeof(res_img)+ sys.getsizeof(range_map)))
    print('time spend in compression: {}s'.format(tend-tini))
    f = open(result_path, 'wb')
    pickle.dump((res_img, range_map),f, -1)
    f.close()
    print('original: {}, encoded: {}'.format(image_path, result_path))

def show_pkl_image(path):
    f = open(path, 'rb')
    im = pickle.load(f)
    f.close()
    im.show(path)
    im = im.convert("RGB")
    im.save('a.BMP')

def fractal(im, result_file,  threshold=35, overlaping=True):
    if overlaping:
        domain_blocks = decomp_overlaping_blocks(im, 16)
    else:
        domain_blocks = decomp_blocks(im, 16)
    range_blocks = decomp_blocks(im, 8)
    range_mapping = {}
    result_img = np.array([])
    #for each range
    for i in range(len(range_blocks)):
        fil = np.array([])
        print(i)
        for j in range(len(range_blocks[0])):
            range_act = range_blocks[i][j]
            # buscar un domain block q sigui prou similar
            block_found = False
            d_i = 0
            while d_i < len(domain_blocks) and not block_found:
                d_j = 0
                while d_j < len(domain_blocks[0]) and not block_found:
                    if not block_found:
                        domain_act = reduce_matrix(np.asarray(domain_blocks[d_i][d_j],dtype='float32'))
                        res = check_transforms(range_act, domain_act, threshold)
                        if res != -1:
                            range_mapping[(i,j)] = res
                 #           print ('block {},{} transformed: ({},{},{})'.format(i,j,d_i,d_j, res))
                            fil=np.append(fil, undo_trans(domain_act, res))
                            block_found = True
                    d_j += 1
                d_i += 1
            if not block_found:
                range_mapping[(i,j)] = "ABCDFASDFA"
                fil = np.append(fil, range_act)
                print("pinx")
        result_img = np.append(result_img, fil)
    result_img = result_img.reshape((64,64,8,8))
    return (result_img, range_mapping)

def decode_image(filename, file_result):
    f = open(filename, 'rb')
    result_img, range_mapping = pickle.load(f)
    f.close()
    tini = time.clock()
    dec = decode(result_img, range_mapping)
    img_d = asImage(dec)
    img_d = img_d.convert("RGB")
    tend = time.clock()
    print('time spend in decompression: {}s'.format(tend-tini))
    img_d.save(file_result)
    return asImage(dec)


if __name__ == '__main__':
    usage = 'python3 fractal-py [-e|-d] FILE1 FILE2'
    parser = OptionParser(usage=usage, version='fractal.py 1.0')
    parser.add_option('-e', '--encode', dest='encode',default=False, action='store_true', help='encode FILE1 into FILE2')
    parser.add_option('-d', '--decode', dest='decode',default=False, action='store_true', help='decode FILE1 into FILE2')
    (options, args) = parser.parse_args()
    
    if options.encode and options.decode:
        print ('Select only one from encode/decode!')
        exit()
    if (not options.encode) and (not options.decode):
        print ('Select at least one from encode/decode')
        exit()
    if len(args) != 2:
        print ('FILE1 and FILE2 required (see -h for more details)')
        exit()
    if options.encode:
        encode_image(args[0], args[1])
    elif options.decode:
        decode_image(args[0], args[1])
    print('end') 
