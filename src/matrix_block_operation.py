from PIL import Image
import numpy as np

image_path = '../samples/lena.bmp'

def decomp_domain_blocks(image, block_size):
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
    return np.asarray(domain_blocks) 


def get_range_block(image, p, w,h):
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

    

def main():
    im = Image.open(image_path)
    print (im.size)
    b = decomp_domain_blocks(im , 8)
    print(len(b), len(b[0]))
    print(np.asarray(im)[0][0:8], np.asarray(im)[1][0:8])
    print(b[0][0])
    b1 = get_range_block(im, (0,0), 2,4)
    print(b1)

if __name__ == '__main__':
    main()
