from PIL import Image
import numpy as np
import os,sys
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
            #new_mat[i][j] = ((mat[2*i][2*j]+mat[2*i+1][2*j+1])/n)
    return np.asarray(new_mat)



def check_transforms(rangeb, domainb, threshold):
    # print(similarity(rangeb, rotate90(domainb)))
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
    elif similarity(rangeb, brightness(domainb,0.15)) <= threshold:
        return 'b15'
    elif similarity(rangeb, brightness(domainb,0.2)) <= threshold:
        return 'b20'
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
    elif trans == 'b15':
        return brightness(block,0.15)
    elif trans == 'b20':
        return brightness(block,0.2)
    else:
        return block

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
    res_img, range_map = fractal(im, result_path, 35,False)
    f = open(result_path, 'wb')
    pickle.dump((res_img, range_map),f, -1)
    f.close()
    print('original: {}, encoded: {}'.format(image_path, result_path))
   
    
    
    #im_reconstruct.save(result_path)

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
    # print((len(range_blocks), len(range_blocks[0]), len(range_blocks[0][0]), len(range_blocks[0][0][0])))
    # im2 = reconstruct_img(range_blocks)
    # asImage(im2).show()
    # print(im2.shape)
    # im2 = asImage(reconstruct_img(range_blocks))
    # print(im2.size)
    # im2.show()
    range_mapping = {}
    result_img = np.array([])
    #for each range
    for i in range(len(range_blocks)):
        fil = np.array([])
        print(i)
        for j in range(len(range_blocks[0])):
            range_act = range_blocks[i][j]
            ### buscar un domain block
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
    #f = open(result_file, 'wb')
    #pickle.dump((result_img, range_mapping), f, -1)
    #f.close()

def decode_image(filename, file_result):
    f = open(filename, 'rb')
    result_img, range_mapping = pickle.load(f)
    f.close()
    dec = decode(result_img, range_mapping)
    # print(np.asarray(im))
    # print(range_mapping)
    # print(dec)
    # print(im.size)
    # print(dec.shape)
    img_d = asImage(dec)
    img_d = img_d.convert("RGB")
    img_d.save(file_result)
    return asImage(dec)
    # res_img = reconstruct_img(result_img)
    # res_img = asImage(res_img)
    # res_img.show()


def main():
    image_path = '../samples/lena.jpg'
    image_path = '../samples/chess.jpg' # funciona mu be amb aqsta XD
    encode_image(image_path, '../samples/chess_compress.bmp')
#    show_pkl_image('../samples/result_overlap.pkl')
#    show_pkl_image('../samples/result.pkl')
    Image.open(image_path).show()
#    show_pkl_image('../samples/chess.pkl')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--encode', dest='encode', action='store_true')
    parser.add_option('-d', '--decode', dest='decode', action='store_true')
    (options, args) = parser.parse_args()
    print(options,'\n', args)
    if options.encode and options.decode:
        print ('Select only one from encode/decode!')
        exit()
    if options.encode:
        encode_image(args[0], args[1])
    elif options.decode:
        decode_image(args[0], args[1])
    #main()
