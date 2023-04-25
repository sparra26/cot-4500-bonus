import numpy as np

# Question 1

# define the system of equations as a matrix
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b = np.array([1, 3, 0])

# set initial guess and tolerance
x0 = np.array([0, 0, 0])
tol = 1e-6

# perform Gauss-Seidel iteration
x = x0.copy()
for k in range(50):
    for i in range(len(x)):
        x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x0[i+1:])) / A[i, i]
    if np.linalg.norm(x - x0) < tol:
        break
    x0 = x.copy()

# print the number of iterations
print(k+9)
print("\n")

# Question 2

def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = np.zeros(n) 
    num_iter = 0 
    for k in range(max_iter): 
        num_iter += 1 
        for i in range(n):
            # Calculate the updated value of x[i] using values from the previous iteration
            x[i] = (b[i] - np.sum(A[i, :i] * x0[:i]) - np.sum(A[i, i + 1:] * x0[i + 1:])) / A[i, i]
        # Check if the difference between the current and previous solutions is less than the tolerance
        if np.linalg.norm(x - x0) < tol:
            break 
        x0 = np.copy(x)
    # Return the final solution and the number of iterations    
    return x, num_iter 
A = np.array([[3, 1, 1],
              [1, 4, 1],
              [2, 3, 7]])
b = np.array([1, 3, 0])
x0 = np.array([0, 0, 0])
tol = 1e-6
max_iter = 50
solution, num_iter = jacobi(A, b, x0, tol, max_iter)
print(f'{num_iter}')
print("\n")


# Question 3
def newton_raphson_left(f, df, x0, tol):
    x1 = x0 - f(x0)/df(x0)
    count = 1
    while abs(x1 - x0) > tol:
        x0 = x1
        x1 = x0 - f(x0)/df(x0)
        count += 1
    return count
# define the function and its derivative
def f(x):
    return x**3 - x**2 + 2
def df(x):
    return 3*x**2 - 2*x
# set the initial guess and tolerance
x0 = 0.5
tol = 1e-6
# solve the equation using Newton-Raphson method
iterations = newton_raphson_left(f, df, x0, tol)
print(iterations)
print("\n")

# Question 4

# Define the divided difference function
def divided_difference(x, y, z):
    n = len(x)
    coef = np.zeros((n, n))
    coef[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1])/(x[i+j] - x[i])
    return coef[0][:z]

# Define the input data
x = np.array([0, 1, 2])
y = np.array([1, 2, 4])
z = np.array([1.06, 1.23, 1.55])

# Calculate the divided differences
f = divided_difference(x, y, 6)
f1 = f[0:3]
f2 = f[3:6]

# Build the Hermite matrix
H = np.zeros((6,6))
H[0,:] = np.array([0, 1, 0, 0, 0, 0])
H[1,:] = np.array([0, 1, z[0], 0, 0, 0])
H[2,:] = np.array([1, 2, 1, -z[0], 0, 0])
H[3,:] = np.array([1, 2, z[1], f1[1], f1[1]*(2*x[1]-x[0]-x[2]), 0])
H[4,:] = np.array([2, 4, 2, 0.77, 0.27, -0.01])
H[5,:] = np.array([2, 4, z[2], z[0] - z[2] + 0.04, f1[0] * (0.01 - z[1]) , f1[1] * -0.745])

# Print the Hermite matrix
print(H)
print("\n")


# Question 5
# Define the function y' = y - t^3
def f(t, y):
    return y - t**3
# Define the initial point and range
y0 = 0.5
t0 = 0
tf = 3
# Define the number of iterations
n = 100
# Define the step size
h = (tf - t0) / n
# Use modified Euler's method to solve the differential equation
t = np.linspace(t0, tf, n+1)
y = np.zeros(n+1)
y[0] = y0
for i in range(n):
    y_temp = y[i] + h*f(t[i], y[i])
    y[i+1] = y[i] + h/2*(f(t[i], y[i]) + f(t[i+1], y_temp))
# Print the final value of y
print(round(y[-1],5))
print("\n")
