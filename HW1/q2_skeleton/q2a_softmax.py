import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    x = x.astype(float)

    if len(x.shape) > 1:
        # Matrix
        x -= x.max(axis=1, keepdims=True)
        np.exp(x, out=x)  # Exp in-place
        x /= x.sum(axis=1, keepdims=True)  # Divide by sum
    else:
        # Vector
        np.exp(x, out=x) # Exp in-place
        x /= np.sum(x)   # Divide by sum

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")

    # ---------------- Test 1: simple vector ----------------
    x = np.array([1.0, 2.0, 3.0])
    y = x.copy()
    assert np.allclose(softmax(y), [0.09003057, 0.24472847, 0.66524096])


    # ---------------- Test 2: simple matrix (row-wise) ----------------
    x = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    y = x.copy()
    softmax(y)
    assert np.allclose(softmax(y), [
        [0.09003057, 0.24472847, 0.66524096],
        [0.09003057, 0.24472847, 0.66524096]
    ])


    # ---------------- Test 3: uniform vector ----------------
    x = np.array([5.0, 5.0, 5.0])
    y = x.copy()
    assert np.allclose(softmax(y), [1/3, 1/3, 1/3])


    # ---------------- Test 4: negative vector ----------------
    x = np.array([-1.0, -2.0, -3.0])
    y = x.copy()
    assert np.allclose(softmax(y), [0.66524096, 0.24472847, 0.09003057])


    # ---------------- Test 5: uniform rows ----------------
    x = np.array([[2.0, 2.0, 2.0],
                [9.0, 9.0, 9.0]])
    y = x.copy()
    assert np.allclose(softmax(y), [
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3]
    ])

    # ---------------- Test 6: random row ----------------
    x = np.array([0.0, 1.0, -1.0])
    y = x.copy()
    assert np.allclose(softmax(y), [0.24472847, 0.66524096, 0.09003057])

    # ---------------- Test 7: 1×N treated as row ----------------
    x = np.array([[0.0, 1.0, 2.0]])
    y = x.copy()
    assert np.allclose(softmax(y), [[0.09003057, 0.24472847, 0.66524096]])

    # ---------------- Test 8: N×1 treated as column-vector rows ----------------
    x = np.array([[1.0],
                [2.0],
                [3.0]])
    y = x.copy()
    assert np.allclose(softmax(y), [[1.0], [1.0], [1.0]])

    # ---------------- Test 9: mixed ints + floats ----------------
    x = np.array([[1, 2.0, 3],
                [0, -1.0, -2]])
    y = x.copy()
    assert np.allclose(softmax(y), [
        [0.09003057, 0.24472847, 0.66524096],
        [0.66524096, 0.24472847, 0.09003057]
    ])


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
