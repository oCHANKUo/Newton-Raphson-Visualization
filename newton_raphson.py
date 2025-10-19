import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Get function input from user
user_input = input("Enter your function in terms of x (e.g., x**3 - 4*x - 9): ")
x = sp.Symbol('x')
f = sp.sympify(user_input)

# derivate
df = sp.diff(f, x)

f_num = sp.lambdify(x, f, "numpy") # need to convert the input into numerical values
df_num = sp.lambdify(x, df, "numpy")

x0 = float(input("Enter initial guess: "))
tolerance = 1e-6
max_iter = 50
iterations = 0
x_values = [x0]  # for visualization

for i in range(max_iter):
    fx = f_num(x0)
    dfx = df_num(x0)
    if dfx == 0:
        print("Derivative became zero. No convergence.")
        break
    x1 = x0 - fx/dfx
    x_values.append(x1)
    iterations += 1
    if abs(x1 - x0) < tolerance:
        print(f"\nRoot found: {x1:.6f}")
        print(f"Iterations: {iterations}")
        break
    x0 = x1
else:
    print("\nDid not converge within max iterations.")

# Visualization
x_range = np.linspace(-5, 5, 100)
y_range = f_num(x_range)

plt.figure(figsize=(8,6))
plt.axhline(0, color='black', linewidth=1)
plt.plot(x_range, y_range, label='f(x)', color='blue')

for i in range(len(x_values)-1):
    plt.plot(x_values[i], f_num(x_values[i]), 'ro')
    plt.plot([x_values[i], x_values[i]], [0, f_num(x_values[i])], 'r--')

plt.title("Newton-Raphson Method: Function & Iteration Evolution")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
