import random

import numpy as np
from numpy.testing import assert_allclose


def gradcheck_naive(f, x, gradient_text=""):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        # Save the old value
        old_val = x[ix]

        # Compute f(x + h)
        x[ix] = old_val + h
        random.setstate(rndstate)
        fxh_plus, _ = f(x)

        # Compute f(x - h)
        x[ix] = old_val - h
        random.setstate(rndstate)
        fxh_minus, _ = f(x)

        # Restore the value
        x[ix] = old_val

        # Centered difference
        numgrad = (fxh_plus - fxh_minus) / (2 * h)

        # Compare gradients
        assert_allclose(numgrad, grad[ix], rtol=1e-5,
                        err_msg=f"Gradient check failed for {gradient_text}.\n"
                                f"First gradient error found at index {ix} in the vector of gradients\n"
                                f"Your gradient: {grad[ix]} \t Numerical gradient: {numgrad}")

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def test_gradcheck_basic():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), 2*x)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))       # scalar test
    gradcheck_naive(quad, np.random.randn(3,))     # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print()


def your_gradcheck_test():
    """
    Additional sanity checks for gradient checking.
    """
    print("Running your sanity checks...")

    # 1. Cubic function
    cubic = lambda x: (np.sum(x ** 3), 3 * x ** 2)
    gradcheck_naive(cubic, np.array(2.0))
    gradcheck_naive(cubic, np.random.randn(5,))
    gradcheck_naive(cubic, np.random.randn(3, 2))

    # 2. Sine function
    sine_func = lambda x: (np.sum(np.sin(x)), np.cos(x))
    gradcheck_naive(sine_func, np.array(0.5))
    gradcheck_naive(sine_func, np.random.randn(4,))
    gradcheck_naive(sine_func, np.random.randn(2, 3))

    # 3. Exponential function
    exp_func = lambda x: (np.sum(np.exp(x)), np.exp(x))
    gradcheck_naive(exp_func, np.array(0.0))
    gradcheck_naive(exp_func, np.random.randn(6,))

    # 4. Logarithm (avoid zero or negative)
    log_func = lambda x: (np.sum(np.log(x)), 1.0 / x)
    gradcheck_naive(log_func, np.array([0.1, 1.0, 2.0]))
    gradcheck_naive(log_func, np.random.rand(3, 3) + 0.1)  # add 0.1 to avoid log(0)

    # 5. Function with randomness (gradient should not depend on random part)
    def random_func(x):
        val = np.sum(x**2) + random.random() * 0.0  # randomness has zero effect
        grad = 2 * x
        return val, grad

    gradcheck_naive(random_func, np.random.randn(4,))
    gradcheck_naive(random_func, np.random.randn(2, 2))

    print("All additional sanity checks passed!\n")


if __name__ == "__main__":
    test_gradcheck_basic()
    your_gradcheck_test()
