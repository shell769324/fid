from matplotlib.image import imread
import numpy as np
import scipy.signal
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.feature import match_template

def getSubImg(img, size, cx, cy):
    halfSize = int(size / 2)
    (lx, ly) = (cx - halfSize, cy - halfSize)
    (rx, ry) = (cx + halfSize + 1, cy + halfSize + 1)
    return img[ly:ry, lx:rx]

def ac(img, size, cx, cy):
    patch = getSubImg(img, size, cx, cy)
    surround = getSubImg(img, size * 3 - 2, cx, cy)
    return match_template(surround, patch)


def acAvg(img, cx, cy, step = 4, start = 11, end = 35):
    accFreq = np.ones((start * 2 - 1, start * 2 - 1))
    accCorr = ac(img, start, cx, cy)
    for i in range(start + step, end + 1, step):
        freq = np.ones((i * 2 - 1, i * 2 - 1))
        freq[step:(step + len(accFreq)), step:(step + len(accFreq))] += accFreq
        accFreq = freq
        corr = ac(img, i, cx, cy)
        corr[step:(step + len(accCorr)), step:(step + len(accCorr))] += accCorr
        accCorr = corr
    center = int(len(accCorr)/2)
    accCorr[center - 2:center + 3, center - 2:center + 3] = np.zeros((5, 5))
    return np.divide(accCorr, accFreq)

def findPeaks(img, numPeaks, thresholdFrac=0.5, neighSpan=1):
    threshold = (thresholdFrac * (img.max() - img.min())) + img.min()

    rows, cols = np.array(np.where(img >= threshold))
    values = []
    for i, j in zip(rows, cols):
        values.append((i, j, img[i, j]))

    dtype = [('row', int), ('col', int), ('intensity', np.float64)]
    indices = np.array(values, dtype=dtype)

    # sort in-place in descending order
    indices[::-1].sort(order='intensity')
    res = []
    # Perform suppression
    for idx in indices:
        intensity = idx[2]
        if intensity <= -1:
            continue

        x0 = idx[1] - neighSpan
        xend = idx[1] + neighSpan
        y0 = idx[0] - neighSpan
        yend = idx[0] + neighSpan

        toSuppress = np.where((indices['col'] >= x0) &
                                       (indices['col'] <= xend) &
                                       (indices['row'] >= y0) &
                                       (indices['row'] <= yend))
        if toSuppress:
            indices['intensity'][toSuppress] = -2
        idx[2] = intensity
        res.append([idx[0], idx[1]])
        if(len(res) >= numPeaks):
            break
    res = np.array(res)
    return res


def isParallel(v1, v2):
    return abs(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) > 0.8

def frac(a):
    return abs(a - round(a))

def h(cpq, peaks, AC, converted, i, j):
    res = 0
    for k in range(len(peaks)):
        if k == i or k == j:
            continue
        abk = cpq.dot(converted[k])
        res += (1 - 2 * max(frac(abk[0]), frac(abk[1]))) * AC[peaks[k][0]][peaks[k][1]]
    return res

def score(peaks, AC, alpha = 2):
    best = -100000
    converted = (peaks - int(len(AC) / 2)).astype(float)
    finalCp = np.array([0, 0])
    finalCq = np.array([0, 0])
    for i in range(len(peaks)):
        cp = peaks[i]
        cpc = converted[i]
        for j in range(i + 1, len(peaks)):
            cq = peaks[j]
            cqc = converted[j]
            cpq = np.array([[cpc[0], cqc[0]],
                            [cpc[1], cqc[1]]])
            if isParallel(cpc, cqc):
                continue
            cpq = np.linalg.inv(cpq)
            local = h(cpq, peaks, AC, converted, i, j)
            local /= np.linalg.norm(cpc) + np.linalg.norm(cqc)
            local += alpha * (AC[cp[0]][cp[1]] + AC[cq[0]][cq[1]])
            if local > best:
                finalCp = cpc
                finalCq = cqc
                best = local
    return best, finalCp, finalCq

def visualizePeaks(peaks, x):
    grid = np.zeros((x, x))
    for peak in peaks:
        grid[peak[0]][peak[1]] = 1
    plt.imshow(grid)
    plt.show()

def ranImg(size):
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            img[i][j] += random.randint(-5, 5)
    return img

def barImg(size):
    img = np.zeros((size, size))
    for i in range(0, size, 2):
        for j in range(size):
            img[i][j] = 1
    return img

def beleaguerImg(size):
    img = np.zeros((size, size))
    for i in range(0, size - 5, 5):
        for j in range(0, size - 5, 5):
            img[i + 5][j + 5] = random.randint(0, 2)
    return img

def footPrintImg():
    img = imread('FID-300/references/00033.png')
    return img


def showImg(img, x, y):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)

    circ = Circle((x, y), 5, color="red")
    ax.add_patch(circ)
    # Show the image
    plt.show()


def showDir(img, v1, v2, cx, cy):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    ax.imshow(img)

    plt.plot([cx, cx + v2[1]], [cy, cy + v2[0]], 'ro-')
    plt.plot([cx, cx + v1[1]], [cy, cy + v1[0]], 'ro-')
    # Show the image
    plt.show()

def demo():
    # img = np.arange(0, 121, 1).reshape((11, 11))
    (cx, cy) = (90, 110)
    (start, end) = (23, 55)
    skip = 6
    size = 60
    img = footPrintImg()

    ac = acAvg(img, cx, cy, skip, start, end)
    showImg(img, cx, cy)
    plt.imshow(ac)
    plt.show()
    peaks = findPeaks(ac, size**2, 0.8, 6)
    (sc, v1, v2) = score(peaks, ac)
    visualizePeaks(peaks, end * 2 - 1)
    print(sc, v1, v2)
    showDir(img, v1, v2, cx, cy)


demo()