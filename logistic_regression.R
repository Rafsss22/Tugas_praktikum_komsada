# Newton-Raphson Method for Logistic Regression
newton_raphson <- function(X, y, tol = 1e-6, max_iter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p) # Initialize beta

  for (iter in 1:max_iter) {
    # Predicted probabilities
    p_hat <- 1 / (1 + exp(-X %*% beta))

    # Gradient (first derivative of log-likelihood)
    grad <- t(X) %*% (y - p_hat)

    # Hessian (second derivative of log-likelihood)
    H <- -t(X) %*% diag(as.vector(p_hat * (1 - p_hat))) %*% X

    # Update beta using Newton-Raphson
    beta_new <- beta - solve(H) %*% grad

    # Check for convergence
    if (sum(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  list(method = "Newton-Raphson", beta = beta, fit = p_hat)
}
# IRLS Method for Logistic Regression
irls <- function(X, y, tol = 1e-6, max_iter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p) # Initialize beta

  for (iter in 1:max_iter) {
    # Predicted probabilities
    p_hat <- 1 / (1 + exp(-X %*% beta))

    # Weight matrix (diagonal of p_hat * (1 - p_hat))
    W <- diag(as.vector(p_hat * (1 - p_hat)))

    # Gradient (first derivative of log-likelihood)
    grad <- t(X) %*% (y - p_hat)

    # Hessian (second derivative of log-likelihood)
    H <- t(X) %*% W %*% X

    # Update beta using IRLS
    beta_new <- beta + solve(H) %*% grad

    # Check for convergence
    if (sum(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  list(method = "IRLS", beta = beta, fit = p_hat)
}
# Simulate data for testing
set.seed(42)
n <- 100
X <- cbind(1, matrix(rnorm(n * 2), nrow = n))  # Adding intercept column
beta_true <- c(0.5, -1, 2)  # True coefficients
y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta_true)))  # Generating binary response

# Run Newton-Raphson and IRLS
newton_result <- newton_raphson(X, y)
irls_result <- irls(X, y)

# Display results
print(newton_result)
print(irls_result)
