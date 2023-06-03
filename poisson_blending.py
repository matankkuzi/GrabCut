import cv2
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
from scipy.signal import convolve2d


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(1, -1)
    mat_D.setdiag(-4)
    mat_D.setdiag(1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(1, 1 * m)
    mat_A.setdiag(1, -1 * m)

    return mat_A


def get_edge_indices(mask):
    # Pad the mask with zeros
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Define the edge detection kernel
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Convolve the mask with the kernel to count the number of edge neighbors
    neighbor_counts = convolve2d(padded_mask, kernel, mode='same')

    # Find the indices of the edges with no more than three neighbors
    edge_indices = np.argwhere((padded_mask == 1) & (neighbor_counts < 4))

    # Remove the padding from the edge indices
    edge_indices -= 1

    flat_indices = np.ravel_multi_index(np.transpose(edge_indices), mask.shape)

    return flat_indices


def get_odd_vals(diff):
    if diff == 2:
        return [1, 1]
    elif diff == 1:
        return [0, 1]
    elif diff == -1:
        return [0, -1]
    elif diff == -2:
        return [-1, -1]
    else:
        return [0, 0]


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    source = im_src
    target = im_tgt
    mask = im_mask

    padding_y = abs(int(center[1] - (source.shape[0] / 2)))
    padding_x = abs(int(center[0] - (source.shape[1] / 2)))

    diff_y = target.shape[0] - (2 * padding_y + source.shape[0])
    diff_x = target.shape[1] - (2 * padding_x + source.shape[1])

    y_odd = get_odd_vals(diff_y)
    x_odd = get_odd_vals(diff_x)

    pad_src = np.pad(source, (
    (padding_y + y_odd[0], padding_y + y_odd[1]), (padding_x + x_odd[0], padding_x + x_odd[1]), (0, 0)),
                     mode='constant', constant_values=0)
    pad_mask = np.pad(mask,
                      ((padding_y + y_odd[0], padding_y + y_odd[1]), (padding_x + x_odd[0], padding_x + x_odd[1])),
                      mode='constant', constant_values=0)
    source = pad_src
    mask = pad_mask

    y_range, x_range = target.shape[:-1]
    mask[mask != 0] = 1

    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc(copy=False)

    mask_flat = mask.reshape(y_range * x_range)
    src_splitted = [source.reshape((y_range * x_range, 3))[:, i] for i in range(3)]
    tgt_splitted = [target.reshape((y_range * x_range, 3))[:, i] for i in range(3)]
    vector_b_splitted = [laplacian.dot(src) for src in src_splitted]

    indices = np.where(mask_flat == 0)[0]
    edges_indices = get_edge_indices(mask)

    for i in range(3):
        vector_b_splitted[i][mask_flat == 0] = tgt_splitted[i][mask_flat == 0]
        vector_b_splitted[i][edges_indices] = tgt_splitted[i][edges_indices]

    for i in edges_indices:
        mat_A.data[i] = [0] * mat_A.getrow(i).nnz
        mat_A[i, i] = 1

    for i in indices:
        mat_A.data[i] = [0] * mat_A.getrow(i).nnz
        mat_A[i, i] = 1

    laplacian = mat_A.tocsc()
    for channel in range(3):
        vector_b = vector_b_splitted[channel]
        x = spsolve(laplacian, vector_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target[:, :, channel] = x

    im_blend = target

    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
