# File used to generate images for thesis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Plot between 0 and 40 with .001 steps.
x_axis = np.arange(0, 40, 0.001)
# Parameters for distributions.
m1, s1 = 10, 2
m2, s2 = 25, 3*s1
# PDF's
fx = norm.pdf(x_axis,m1, s1)
gx = norm.pdf(x_axis,m2, s2)
hx = norm.pdf(x_axis,m1 + 0.5*m2, s1 + 0.5*s2)
# Show two separate pdf's
plt.subplot(2,1,1)
plt.plot(x_axis, fx, label = "f(x)")
plt.plot(x_axis, gx, label = "g(x)")
plt.legend()

plt.subplot(2,1,2)
plt.plot(x_axis, hx, 'r',label = "h(x)")
plt.legend()
plt.show()
