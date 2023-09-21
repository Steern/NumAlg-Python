def goldstein(f, grad_f, x, p, alpha=1.0, c1=0.1, my = 20, sigma=0.1, max_iter=100):

    alpha_last = 0
    for _ in range(max_iter):
        
        condition1 = f(x + alpha * p) <= f(x) + c1 * alpha * grad_f(x).dot(p)
        if condition1:
            print("first cond linesearch")
            return alpha

        condition2 = grad_f(x + alpha * p).dot(p) >= sigma * grad_f(x).dot(p)
        if condition2:
            print("second cond linesearch")
            return alpha

        if 2*alpha - alpha_last >= my:
            return my
        
        tmp_alpha = alpha
        alpha = 2*alpha - alpha_last
        alpha_last = tmp_alpha

    print(f"max_iter reached in linesearch")    
    return alpha