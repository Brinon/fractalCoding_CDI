import numpy as np
'''
Range blocks -> non overlaping, NxN
Domain blocks -> may be overlaping, N'x N', N' > N
'''
def reconstruct_img(blocks):
    ''' blocks is a N*N square matrix '''
    res = []
    for row in blocks:
        # row array de blocks quadrats
        for fila in range(len(row[0])):
            r = np.asarray([])
            for b in range(len(row)):
                # print(b,fila)
                r = np.append(r, row[b][fila])
            res.append(r)
    return np.vstack(res)

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

def rotate90(mat):
    return np.rot90(mat,-1)

def rotate180(mat):
    return np.rot90(mat,2)

def rotate270(mat):
    return np.rot90(mat,1)

def flip_vertical(mat):
    return np.flipud(mat)

def flip_horizontal(mat):
    return np.fliplr(mat)

def similarity(m1, m2):
    sim = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            sim += abs(m1[i][j] - m2[i][j])
    return sim/(len(m1)*len(m1[0]))



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

