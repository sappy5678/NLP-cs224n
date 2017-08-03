import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """

    # by 公式  http://i.imgur.com/KwB9HrK.png
    # 參考資料
    #  softmax
    # https://www.zhihu.com/question/23765351
    # https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        # axis 是因為要對每一行分別做計算，像是把 matrix 轉成很多個 vector 一樣\
        # 取max 是 因為要讓數字不要太大，而且相減後做除法的質也不會改變
        max = np.max(x,axis=1)
        # 轉成可以對每一行正確相減的格式
        max = max.reshape((x.shape[0], 1))
        x = x - max
        x_exp = np.exp(x)
        # axis 是因為要對每一行分別做計算，像是把 matrix 轉成很多個 vector 一樣
        x_sum = np.sum(x_exp,axis=1)
        x = x_exp / x_sum
        # raise NotImplementedError
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        # 取max 是 因為要讓數字不要太大，而且相減後做除法的質也不會改變
        max = np.max(x)
        x = x - max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        x = x_exp / x_sum
        # raise NotImplementedError
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    pass
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
