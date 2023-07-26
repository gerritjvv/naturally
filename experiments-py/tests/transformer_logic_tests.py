import timeit

import numpy as np


def apply_softmax(W_prime: np.ndarray):
    #
    # For softmax we have
    # for each row n where each column c in row n  is e^c
    # then each row's columns are summed
    # then each row n where each column c is divided by the sum of each row in the previous step
    #
    # To do this efficiently in matrices we can do:
    # Apply the exponential function to each element in the matrix
    # Create a matrix v_p2 = row sums using np.sum( v_p1, axis=1 ) axis=1 means sum each row
    # Then divide the v_p1 by the column vector of v_p2. Using a column vector means the
    #  each row in v_p1 is divided by the only column in the matching row in v_p2.
    #
    # e.g.
    # we have v_p1
    # | e^a  e^b |
    # | e^c  e^d |
    #
    # we also want vp_2
    # | e^a + e^b |   as a column vector | [ e^a + e^b ] |
    # | e^c + e^d |                      | [ e^c + e^d ] |
    #
    # Then we divide vp_1 / vp_2
    #
    # | e^a / (e^a + e^b),   e^b / (e^a + e^b) |
    # | e^c / (e^c + e^d),   e^d / (e^c + e^d) |
    #
    #
    #  if done correctly the sum of the columns for any row should be 1
    #  print(f"v_p3: {np.sum(v_p3, axis=1)}")

    v_p1 = np.exp(W_prime)
    v_p2 = np.sum(v_p1, axis=1)
    v_p2 = v_p2.reshape((v_p2.shape[0], 1))  # create a column vector

    # print("---------- softmax ------------")
    # print(f"v_p1: {v_p1}")
    # print()
    # print(f"v_p2: {v_p2}")
    # print("---------- softmax ------------")

    v_p3 = v_p1 / v_p2

    return v_p3


def perf_scipy_soft_max(x):
    from scipy.special import softmax

    softmax(x, axis=1)


def test_attention_logic():
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])

    print()
    print(f"x: {x}")
    print(f"x.T: {x.T}")

    # should result in
    #  | 1 2 3 |     | 1 4 |
    #  | 4 5 6 |  .  | 2 5 |
    #                | 3 6 |
    #
    #  | 1*1 + 2*2 + 3*3,  1*4+ 2*5 + 3*6  |       | 14   32 |
    #  | 1*4+ 2*5 + 3*6    4*4 + 5*5 + 6*6 |    => | 32   77 |
    #
    W_prime = x.dot(x.T)

    assert (W_prime == np.array([[14, 32], [32, 77]])).all()

    print(f"W_prime => {W_prime}")
    print(f"{x.dot(x.T)}")

    W = apply_softmax(W_prime)

    # after applying the softmax we must have each row's sum equal to one
    assert (np.sum(W, axis=1) == 1).all()

    print(f"W => {W}")

    from scipy.special import softmax

    print(f"Scipy softmax: {softmax(W_prime, axis=1)}")


def perf_test():
    import math

    # x = np.array([[1, 0.5, 0.2, 3],
    #               [1, -1, 7, 3],
    #               [2, 12, 13, 3]])

    x = np.random.rand(10000, 10000)

    loop = 5

    result = timeit.timeit(lambda: perf_scipy_soft_max(x), number=loop)
    scipy_res = result / loop
    print(f"scipy: {scipy_res}")

    result = timeit.timeit(lambda: apply_softmax(x), number=loop)
    our_max_res = result/loop
    print(f"our_softmax: {float(our_max_res)}")


    if scipy_res < our_max_res:
        print(f"Scripy win {our_max_res - scipy_res}")
    else:
        print(f"Our max win {scipy_res - our_max_res}")
    # 0.0002666301020071842


if __name__ == "__main__":
    perf_test()
