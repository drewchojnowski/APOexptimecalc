
#Integration module for exposure time calc
#An integration function exists in scipy.integrate

#This function is basically a crappy version of riemann sums.
def int(func,a,b):
    #func = function to be integrated. a = lower bound, b = upper bound
    n = 10000000 #number of rectangles to sum. Can make this user input
    x = (b-a)/n #the width of a rectangle (determined by bounds and user-input n)
    m = 0
    for i in range(0,n):
        m += func((x/2.) + (i*x))*x
    return m

