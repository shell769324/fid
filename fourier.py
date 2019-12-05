import cv2
import numpy as np
import csv
import glob
import sys
from matplotlib import pyplot as plt
from scipy import signal
import random
from skimage.feature import match_template

from matplotlib.patches import Circle
from matplotlib.image import imread

from DLL import LinkedList, Node
from autocorrelation import getSample, getSampleWith, getSampleWithMult


def rect(X, v1, v2, img):
    """
    Finds bbox.

    Parameters:
    X(tuple): coordinates of origin.
    v1: x basis vector.
    v2: y basis vector.
    img(2d array): image.

    Returns:
    2d array(img): bbox centered at X, bounding (x +/- v1) & (x +/- v2)
    """
    add_v1 = np.add(X, v1)
    sub_v1 = np.subtract(X, v1)
    add_v2 = np.add(X, v2)
    sub_v2 = np.subtract(X, v2)
    rows = [add_v1[0], sub_v1[0], add_v2[0], sub_v2[0]]
    cols = [add_v1[1], sub_v1[1], add_v2[1], sub_v2[1]]
    max_row = max(rows) + 1
    max_col = max(cols) + 1
    min_row = min(rows)
    min_col = min(cols)
    #print((max_row, min_row, max_col, min_col))
    # Clamping the max/in to get the real input patch
    height = img.shape[0]
    width = img.shape[1]
    max_row_real = np.clip(max_row, 0, height)
    min_row_real = np.clip(min_row, 0, height - 1)
    max_col_real = np.clip(max_col, 0, width)
    min_col_real = np.clip(min_col, 0, width - 1)
    clipped = img[min_row_real:max_row_real, min_col_real:max_col_real]
    # Check if the bbox is out of bound
    if (max_row != max_row_real or max_col != max_col_real or
        min_row != min_row_real or min_col != min_col_real):
        # Pad the out of bound area with random numbers
        pad_row_num = max_row - min_row
        pad_col_num = max_col - min_col
        pad = np.random.rand(pad_row_num, pad_col_num)
        row_offset = min_row_real - min_row
        col_offset = min_col_real - min_col
        pad[row_offset:row_offset + len(clipped),
            col_offset:col_offset + len(clipped[0])] = clipped
        return pad if (clipped.shape[0] * clipped.shape[1] >
                    0.7 * pad.shape[0] * pad.shape[1]) else np.array([])
    return clipped


def compute_four_descriptor(x, v1, v2, img):
    """
    Return Fourier Transformed Image.

    Parameters:
    x(tuple): coordinates of origin.
    v1/height: x basis magnitude.
    v2/width: y basis magnitude.
    img(2d array): image.

    Returns:
    2d array(img): magnitude spectrum of the fourier transform for img
    """
    # Crop out a bbox
    patch = rect(x, v1, v2, img)
    if len(patch) == 0:
        return []
    # plt.imshow(patch, cmap='gray')
    # plt.title('patch')
    # plt.show()
    # Create Gaussian Window
    # TODO
    # Compute Fourier Transform
    discrete_ft = cv2.dft(np.float32(patch), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(discrete_ft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # Plotting the transformed img
    #plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.title('Magnitude Spectrum')
    #plt.show()
    return magnitude_spectrum


def print_ll(ll):
    head = ll.front.next
    while head != ll.back:
        head = head.next


def preprocess_H(H, avg):
    """
    Reorganize the information stored in H.

    Parameters:
    H :[[(pixel 00 info), (pixel_01 info)],
        [(pixel 10 info), (pixel_11 info)]]

    Returns:
    ll: doubly linked list, where ll.front & ll.back are dummy nodes
        The data field of each node is in the form: [coordinates:(i,j), info:H[i,j]],
        where H[i,j] = [h_score(int), v1(list), v2(list)]
    """
    ll = LinkedList()
    nodes = []
    # (1) create a node for each (i,j)th pixel: node([coordinates:(i,j), info:H[i,j]])
    for i in range(len(H)):
        for j in range(len(H[0])):
            new_node = Node([(i, j), H[i][j]])
            # pixel_map[(i, j)] = new_node
            nodes.append(new_node)

    # (2) sort nodes
    nodes.sort(key=lambda x: -x.data[1][0])  # sort all tuple with respect to the h score

    # (3) insert nodes to a linkedlist, where the first element has the highest h_score
    for node in nodes:
        ll.push_back(node)
    #print_ll(ll)
    return ll

def group_patterns(H, img, avg):
    """
    Based on the h_score of each pixel and groups pixels into patterns.

    Parameters:
    H :[[(pixel 00 info), (pixel_01 info)],
        [(pixel 10 info), (pixel_11 info)]]
    img: original image

    Returns:
    patterns(list): stores the data stored in those nodes who are identified as the centers of
                    periodic patterns.
    """
    patterns = []
    info_ll = preprocess_H(H, avg)  # coord_map maps [i,j] to node(H[i,j])
    sim_threshold = 0.44
    s_scores = []
    while info_ll.front.next != info_ll.back:
        # Get the info of the current pixel
        x_node = info_ll.pop_front()
        x_data = x_node.data
        x_coord = x_data[0]
        x_info = x_data[1]
        x_v1 = x_info[1]
        x_v2 = x_info[2]
        f_x = compute_four_descriptor(x_coord, x_v1, x_v2, img)  # f_X is the magnitude spectrum
        if len(f_x) == 0:
            info_ll.delete_node(x_node)
            continue
        # Now, loop through each node (remaining pixel) in ll, compare similarity to cur pixel
        y_node = x_node.next
        while y_node != info_ll.back:
            # Get the info of the other node
            y_data = y_node.data
            y_coord = y_data[0]
            if np.linalg.norm(np.array([y_coord[0], y_coord[1]]) - np.array([x_coord[0], x_coord[1]])) < 35:
                info_ll.delete_node(y_node)
                y_node = y_node.next
                continue
            f_y = compute_four_descriptor(y_coord, x_v1, x_v2, img)
            if len(f_y) == 0:
                info_ll.delete_node(y_node)
                y_node = y_node.next
                continue
            # Check if the similarity between X and Y are high enough
            s = match_template(f_x, f_y)[0][0]
            s_scores.append(s)
            if s > sim_threshold:
                temp = y_node
                info_ll.delete_node(temp)
            y_node = y_node.next
        # add pixel(i,j) to patterns
        patterns.append(x_data)
    #print("Here are the scores")
    #print(s_scores)
    #print("Its average is:")
    #print(np.mean(np.array(s_scores)))
    #print("The patterns are:")
    #print(patterns)
    return patterns

def darn(img, patterns, colors):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(img)
    for i in range(len(patterns)):
        pattern = patterns[i]
        (cx, cy) = (pattern[0][1], pattern[0][0])
        (_, v1, v2) = pattern[1]
        plt.plot([cx, cx + v2[1]], [cy, cy + v2[0]], c=colors[i], linewidth=5)
        plt.plot([cx, cx + v1[1]], [cy, cy + v1[0]], c=colors[i], linewidth=5)
    # Show the image
    plt.show()

def comp2(patterns1, patterns2, img1, img2):
    totalSim = 0
    colors2 = [np.random.rand(3, ) for _ in range(len(patterns2))]
    colors1 = []
    for i in range(len(patterns1)):
        p1 = patterns1[i]
        mag1 = compute_four_descriptor(p1[0], p1[1][1], p1[1][2], img1)
        maxSim = -100
        maxIdx = 0
        for j in range(len(patterns2)):
            p2 = patterns2[j]
            mag2 = compute_four_descriptor(p2[0], p1[1][1], p1[1][2], img2)
            if len(mag2) == 0:
                continue
            thisSim = match_template(mag1, mag2)[0][0]
            if thisSim > maxSim:
                maxSim = thisSim
                maxIdx = j
        colors1.append(colors2[maxIdx])
        totalSim += maxSim
    totalSim /= len(patterns1)
    #darn(img1, patterns1, colors1)
    #darn(img2, patterns2, colors2)
    print("Similarity is", totalSim)
    return totalSim


def match():
    (img1, H, avg) = getSampleWithMult("FID-300/tracks_cropped/00026.jpg")
    patterns1 = group_patterns(H, img1, avg)
    (img2, H, avg) = getSampleWithMult("FID-300/references/00013.png")
    patterns2 = group_patterns(H, img2, avg)
    (img3, H, avg) = getSampleWithMult("FID-300/references/00009.png")
    patterns3 = group_patterns(H, img3, avg)
    comp2(patterns1, patterns2, img1, img2)
    comp2(patterns1, patterns3, img1, img3)

match()

#img = imread('FID-300/references/00033.png')
# compute_four_descriptor([70,70], [20,20], [0, 50], img)
# H = [[(11, [1, 2], [1, 3]), (11, [1, 2], [1, 3])],
#      [(2, [2, 2], [2, 3]), (22, [2, 2], [2, 3])],
#      [(3, [3, 2], [3, 3]), (-3, [3, 2], [3, 3])]]
# print(H)
def test():
    (img, H, avg) = getSample()
    patterns = group_patterns(H, img, avg)

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)
    for pattern in patterns:
        (cx, cy) = (pattern[0][1], pattern[0][0])
        (_, v1, v2) = pattern[1]
        col = np.random.rand(3, )
        plt.plot([cx, cx + v2[1]], [cy, cy + v2[0]], c=col, linewidth=5)
        plt.plot([cx, cx + v1[1]], [cy, cy + v1[0]], c=col, linewidth=5)
    # Show the image
    plt.show()

def runBig():
    ans = dict()
    sys.stdout = open('result', 'w')
    with open('FID-300/label_table.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            ans[int(row[0])] = int(row[1])
    memoiCrime = dict()
    for filename in glob.glob('crime/*.jpg'):  # assuming gif
        last = int(filename[-8:-4])
        (img, H, avg) = getSampleWith(filename)
        patterns = group_patterns(H, img, avg)
        memoiCrime[last] = (patterns, img)

    memoiRef = dict()
    for filename in glob.glob('ref/*.png'):  # assuming gif
        last = int(filename[-8:-4])
        (img, H, avg) = getSampleWith(filename)
        patterns = group_patterns(H, img, avg)
        memoiRef[last] = (patterns, img)

    print("Crime | prediction | similarity")
    for crimeID, crime in memoiCrime.items():
        (patternsC, imgC) = crime
        highSim = 0
        okID = 0
        for refID, ref in memoiRef.items():
            (patternsR, imgR) = ref
            sim = comp2(patternsC, patternsR, imgC, imgR)
            if sim > highSim:
                okID = refID
                highSim = sim
        print(crimeID, okID, highSim)

#runBig()