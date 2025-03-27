logistic_combined <- function(X, y, method = "newton", max_iter = 100, tol = 1e-6) {

  # Menambahkan kolom 1 untuk intersep (bias)
  X <- cbind(1, X)

  # Fungsi Sigmoid
  sigmoid <- function(z) 1 / (1 + exp(-z))

  # Inisialisasi parameter theta
  theta <- rep(0, ncol(X))

  # Fungsi untuk metode Newton-Raphson
  logistic_newton <- function(X, y) {
    p <- sigmoid(X %*% theta)
    gradient <- t(X) %*% (y - p)
    H <- -t(X) %*% diag(as.vector(p * (1 - p))) %*% X
    theta_new <- theta - solve(H) %*% gradient
    return(theta_new)
  }

  # Fungsi untuk metode IRLS
  logistic_irls <- function(X, y) {
    p <- sigmoid(X %*% theta)
    W <- diag(as.vector(p * (1 - p)))
    theta_new <- solve(t(X) %*% W %*% X) %*% t(X) %*% (y - p)
    return(theta_new)
  }

  # Pemilihan metode
  if (method == "newton") {
    for (i in 1:max_iter) {
      # Prediksi probabilitas
      p <- sigmoid(X %*% theta)

      # Regresi Logistik menggunakan Newton-Raphson
      theta_new <- logistic_newton(X, y)

      # Cek konvergensi
      if (max(abs(theta_new - theta)) < tol) {
        break
      }
      theta <- theta_new
    }
  } else if (method == "irls") {
    for (i in 1:max_iter) {
      # Prediksi probabilitas
      p <- sigmoid(X %*% theta)

      # Regresi Logistik menggunakan IRLS
      theta_new <- logistic_irls(X, y)

      # Cek konvergensi
      if (max(abs(theta_new - theta)) < tol) {
        break
      }
      theta <- theta_new
    }
  }

  # Hitung probabilitas (fit)
  fit <- sigmoid(X %*% theta)

  # Output
  result <- list(
    method = method,
    beta = theta,
    fit = fit
  )

  return(result)
}
# Contoh data
set.seed(42)
X <- matrix(rnorm(100), ncol = 2)  # 50 sampel, 2 fitur
y <- sample(c(0, 1), 50, replace = TRUE)

# Menggunakan metode Newton-Raphson
result_newton <- logistic_combined(X, y, method = "newton")
cat("Method:", result_newton$method, "\n")
cat("Beta (Parameter):\n")
print(result_newton$beta)
cat("Fit (Probability):\n")
print(result_newton$fit)

# Menggunakan metode IRLS
result_irls <- logistic_combined(X, y, method = "irls")
cat("\nMethod:", result_irls$method, "\n")
cat("Beta (Parameter):\n")
print(result_irls$beta)
cat("Fit (Probability):\n")
print(result_irls$fit)
