import numpy as np
import cv2
import argparse
import time
import scipy.stats as sc
from sklearn.cluster import KMeans
from collections import Counter

from gmm import GmmComponent
import global_calcs

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

global beta, graph, threshold, prev_energy


# Define the GrabCut algorithm function
def grabcut(img, rect, n_components=5):
    # Initalize global graph and beta
    img = img.astype(np.int16)

    global beta, graph
    beta = global_calcs.calculate_beta(img)
    graph = global_calcs.calculate_graph(img, beta)

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD
    bgGMM, fgGMM = initalize_GMMs(img, mask, n_components)

    global threshold, prev_energy
    threshold = 100
    prev_energy = 0
    energy = 0

    num_iters = 1000

    for i in range(num_iters):
        start_time = time.time()  # record start time
        print(f'(({i})):')
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        prev_energy = energy

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        print(f'energy: {energy}, diff: {prev_energy - energy}')

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

        print(f'iteration time: {time.time() - start_time}')
        print(f'{"-"*50}\n')
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    bgGMM = {}
    fgGMM = {}

    h, w, c = img.shape
    img_flat = img.reshape((h * w, c))
    mask_flat = mask.reshape(h * w)
    print("Initialize GMMs:")
    print(f'GC_BGD: {len(np.where(mask_flat==GC_BGD)[0])} GC_FGD: {len(np.where(mask_flat==GC_BGD)[0])}, GC_PR_BGD: {len(np.where(mask_flat==GC_PR_BGD)[0])}, GC_PR_FGD: {len(np.where(mask_flat==GC_PR_FGD)[0])}')

    bg_indices = np.where((mask_flat == GC_BGD) | (mask_flat == GC_PR_BGD))[0]
    fg_indices = np.where((mask_flat == GC_FGD) | (mask_flat == GC_PR_FGD))[0]

    # Partition pixels into foreground and background
    fg_pixels = img_flat[fg_indices]
    bg_pixels = img_flat[bg_indices]

    fg_kmeans = KMeans(n_clusters=n_components).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components).fit(bg_pixels)

    for i in range(n_components):
        fg_vectors = fg_pixels[fg_kmeans.labels_ == i]
        fgGMM[i] = GmmComponent(fg_vectors, len(fg_pixels))

        bg_vectors = bg_pixels[bg_kmeans.labels_ == i]
        bgGMM[i] = GmmComponent(bg_vectors, len(bg_pixels))

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    h, w, c = img.shape
    mask_flat = mask.reshape(h * w)
    img_flat = img.reshape((h * w, c))


    bg_indices = np.where((mask_flat == GC_BGD) | (mask_flat == GC_PR_BGD))[0]
    fg_indices = np.where((mask_flat == GC_FGD) | (mask_flat == GC_PR_FGD))[0]

    fgPixels = img_flat[fg_indices]
    bgPixels = img_flat[bg_indices]

    n_components = len(bgGMM)

    # unpack the means, covariances, and weights into separate arrays
    bg_labels = global_calcs.calculate_labels_from_gmms(bgGMM, bgPixels)
    fg_labels = global_calcs.calculate_labels_from_gmms(fgGMM, fgPixels)

    bg_counter = []
    fg_counter = []
    for i in range(n_components):
        fgVectors = fgPixels[fg_labels == i]
        fgGMM[i] = GmmComponent(fgVectors, len(fgPixels))

        bgVectors = bgPixels[bg_labels == i]
        bgGMM[i] = GmmComponent(bgVectors, len(bgPixels))
        fg_counter.append(len(fgVectors))
        bg_counter.append(len(bgVectors))

    print(f'fgPixels count: {len(fgPixels)}, bgPixels count: {len(bgPixels)}')
    print(f'foreground label counts: {", ".join(f"{fg_counter[i]}" for i in range(n_components))}')
    print(f'background label counts: {", ".join(f"{bg_counter[i]}" for i in range(n_components))}')

    return bgGMM, fgGMM



def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]

    global graph
    graph2 = global_calcs.add_weighted_t_links_edges_to_graph(graph, img, mask, bgGMM, fgGMM)

    h, w = mask.shape[:2]
    n_nodes = h * w + 2
    source_node = n_nodes - 2  # Back. T-link
    sink_node = n_nodes - 1  # Fore. T-link
    cut = graph2.st_mincut(sink_node, source_node, 'weight')

    fore_cut = cut.partition[0]
    fore_cut.remove(sink_node)

    back_cut = cut.partition[1]
    back_cut.remove(source_node)

    energy = cut.value

    min_cut[0] = fore_cut
    min_cut[1] = back_cut

    return min_cut, energy


def update_mask(mincut_sets, mask):
    h, w = mask.shape
    flat_mask = mask.reshape(h * w)

    fore_cut = mincut_sets[0]
    back_cut = mincut_sets[1]
    print("Update Mask:")
    print(f'GC_BGD: {len(np.where(flat_mask==GC_BGD)[0])} GC_FGD: {len(np.where(flat_mask==GC_FGD)[0])}, GC_PR_BGD: {len(np.where(flat_mask==GC_PR_BGD)[0])}, GC_PR_FGD: {len(np.where(flat_mask==GC_PR_FGD)[0])}')
    for node in fore_cut:
        if flat_mask[node] == GC_BGD:
            flat_mask[node] = GC_PR_FGD

    for node in back_cut:
        if flat_mask[node] == GC_PR_FGD:
            flat_mask[node] = GC_BGD

    print(f'GC_BGD: {len(np.where(flat_mask==GC_BGD)[0])} GC_FGD: {len(np.where(flat_mask==GC_FGD)[0])}, GC_PR_BGD: {len(np.where(flat_mask==GC_PR_BGD)[0])}, GC_PR_FGD: {len(np.where(flat_mask==GC_PR_FGD)[0])}')

    return flat_mask.reshape((h, w))


def check_convergence(energy):
    # TODO: implement convergence check
    global prev_energy, threshold

    if prev_energy == 0:
        prev_energy = energy
        return False

    if prev_energy - energy <= threshold:
        return True

    return False


def cal_metric(predicted_mask, gt_mask):
    sum = np.sum(predicted_mask == gt_mask)
    total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
    acc = sum / total_pixels

    intersection = np.logical_and(predicted_mask, gt_mask).sum()
    union = np.logical_or(predicted_mask, gt_mask).sum()
    jcd= intersection / union

    return (acc,jcd)



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
    print(f'total time: {time.time() - start_time}')

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
