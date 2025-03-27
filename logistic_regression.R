# Newton Raphson
newton_raphson <- function(X, y, tol = 1e-6, max_iter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)

  for (iter in 1:max_iter) {
    p_hat <- 1 / (1 + exp(-X %*% beta))
    grad <- t(X) %*% (y - p_hat)
    H <- -t(X) %*% diag(as.vector(p_hat * (1 - p_hat))) %*% X
    beta_new <- beta - solve(H) %*% grad
    if (sum(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  list(method = "Newton-Raphson", beta = beta, fit = p_hat)
}

# IRLS
irls <- function(X, y, tol = 1e-6, max_iter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)

  for (iter in 1:max_iter) {
    p_hat <- 1 / (1 + exp(-X %*% beta))
    W <- diag(as.vector(p_hat * (1 - p_hat)))
    grad <- t(X) %*% (y - p_hat)
    H <- t(X) %*% W %*% X
    beta_new <- beta + solve(H) %*% grad
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
X <- cbind(1, matrix(rnorm(n * 2), nrow = n))
beta_true <- c(0.5, -1, 2)
y <- rbinom(n, 1, 1 / (1 + exp(-X %*% beta_true)))

# Run Newton-Raphson and IRLS
newton_result <- newton_raphson(X, y)
irls_result <- irls(X, y)

# Display results
print(newton_result)
print(irls_result)
