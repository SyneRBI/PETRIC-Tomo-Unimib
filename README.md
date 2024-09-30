# Preconditioned conjugate gradient descent algorithm
L. Presotto, M. Colombo
Department of physics G. Occhialini, university of Milano Bicocca Milano (MI), Italy

# Algorithm
The proposed algorithm is a classic diagonally preconditioned conjugated gradient descent one.
As the Poisson likelihood diverges in the background, special care is used in computing the diagonal preconditioner, which accounts for both the tomographic part and the prior.

Actually, minor changes in the background area completely change the convergence speed

The second innovation of the algorithm is the use of a way to compute approximately the ideal step size but fastly.
This allows using conjugate gradient descent techniques and to get a quite effective speed up of the convergence

