import numpy as np
import igraph as ig
import scipy.stats as sc
from gmm import GmmComponent

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

global k

def calculate_beta(img):
    sum_total = 0
    diag_down = list((img[i + 1, j + 1] - img[i, j] for i in range(img.shape[0] - 1) for j in range(img.shape[1] - 1)))
    diag_down_array = np.power(np.array(diag_down), 2)
    sum1 = diag_down_array.sum()

    sum_total += sum1

    diag_up = list((img[i - 1, j + 1] - img[i, j] for i in range(1, img.shape[0]) for j in range(img.shape[1] - 1)))
    diag_up_array = np.power(np.array(diag_up), 2)
    sum2 = diag_up_array.sum()
    sum_total += sum2

    result_down = np.diff(img, axis=0) ** 2
    sum_total += result_down.sum()

    result_right = np.diff(img, axis=1) ** 2
    sum_total += result_right.sum()

    cnt = diag_down_array.shape[0] + diag_up_array.shape[0] + (result_down.shape[0] * result_down.shape[1]) + (
            result_right.shape[0] * result_right.shape[1])

    sum_total = ((sum_total / cnt) * 2)

    beta = 1 / sum_total

    return beta


def calculate_graph(img, beta):
    h, w = img.shape[:2]
    n_nodes = h * w + 2  # add source and sink nodes

    graph = ig.Graph(n_nodes, directed=False)
    edge_list = []
    weight_list = []
    for i in range(h):
        for j in range(w):
            pixel_index = i * w + j
            if j + 1 < w:  # right
                weight = calculate_nlink_weight(img[i][j], img[i][j + 1], beta, 1)
                weight_list.append(weight)
                neighbor_index = pixel_index + 1
                edge_list.append((pixel_index, neighbor_index))

            if i + 1 < h:  # down
                weight = calculate_nlink_weight(img[i][j], img[i + 1][j], beta, 1)
                weight_list.append(weight)
                neighbor_index = pixel_index + w
                edge_list.append((pixel_index, neighbor_index))

            if i + 1 < h and j - 1 > 0:  # down left
                weight = calculate_nlink_weight(img[i][j], img[i + 1][j - 1], beta, np.sqrt(2))
                weight_list.append(weight)
                neighbor_index = pixel_index + w - 1
                edge_list.append((pixel_index, neighbor_index))

            if i + 1 < h and j + 1 < w:  # down right
                weight = calculate_nlink_weight(img[i][j], img[i + 1][j + 1], beta, np.sqrt(2))
                weight_list.append(weight)
                neighbor_index = pixel_index + w + 1
                edge_list.append((pixel_index, neighbor_index))

    graph.add_edges(edge_list)
    graph.es['weight'] = weight_list
    global k
    k = 1e300

    return graph


def calculate_nlink_weight(v1, v2, beta, mat_distance):

    return (50 / mat_distance) * (np.exp((-beta) * np.float_power(np.linalg.norm(v1 - v2),2)))


def calculate_labels_from_gmms(gmm: dict[int, GmmComponent], pixels):
    n_components = len(gmm)
    scores = np.array([calculate_score(gmm[i], pixels) for i in range(n_components)])
    labels = np.argmax(scores, axis=0)
    return labels


def add_weighted_t_links_edges_to_graph(graph: ig.Graph, img, mask, bgGMM: dict[int, GmmComponent],
                                        fgGMM: dict[int, GmmComponent]):
    n_component = len(bgGMM)
    h, w, c = img.shape
    n_pixels = h * w
    mask_flat = mask.reshape(n_pixels)
    img_flat = img.reshape((n_pixels, c))
    global k

    n_nodes = n_pixels + 2
    source_node = n_nodes - 2  # Back. T-link
    sink_node = n_nodes - 1  # Fore. T-link

    edge_list = []
    weight_list = []

    pr_indexs = np.where(np.logical_or(mask_flat == GC_PR_BGD, mask_flat == GC_PR_FGD))[0]
    bgd_indexes = np.where(mask_flat == GC_BGD)[0]
    fgd_indexes = np.where(mask_flat == GC_FGD)[0]

    # handle probably edges (GC_PR_..)
    edge_list.extend(list(zip([source_node] * pr_indexs.size, pr_indexs)))
    weight_list.extend(calculate_prob_t_weights(fgGMM, img_flat[pr_indexs]))

    edge_list.extend(list(zip([sink_node] * pr_indexs.size, pr_indexs)))
    weight_list.extend(calculate_prob_t_weights(bgGMM, img_flat[pr_indexs]))


    # handle GC_BGD
    edge_list.extend(list(zip([source_node] * bgd_indexes.size, bgd_indexes)))
    weight_list.extend([np.inf]*bgd_indexes.size)

    edge_list.extend(list(zip([sink_node] * bgd_indexes.size, bgd_indexes)))
    weight_list.extend([0] * bgd_indexes.size)

    # handle GC_FGD
    edge_list.extend(list(zip([source_node] * fgd_indexes.size, fgd_indexes)))
    weight_list.extend([0] * fgd_indexes.size)

    edge_list.extend(list(zip([sink_node] * fgd_indexes.size, fgd_indexes)))
    weight_list.extend([np.inf] * fgd_indexes.size)

    start_index = len(graph.es)
    graph2 = graph.copy()
    graph2.add_edges(edge_list)
    graph2.es[start_index:]['weight'] = weight_list

    return graph2

def calculate_prob_t_weights(gmm: dict[int, GmmComponent], pixels):
    n_components = len(gmm)
    probs = [calculate_score(gmm[i], pixels) for i in range(n_components)]
    return -np.log(np.sum(probs, axis=0))


def calculate_score(gmm:GmmComponent, pixels):
    score = np.zeros(pixels.shape[0])
    if gmm.weight > 0:
        diff = pixels - gmm.mean
        exp = np.exp(-.5 * np.einsum('ij,ij->i', np.matmul(diff, gmm.inverse_covariance_matrix), diff))
        score = exp * gmm.weight * 1 / np.sqrt(gmm.determinant_covariance_matrix)
    return score
