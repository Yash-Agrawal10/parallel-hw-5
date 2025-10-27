#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <omp.h>
#include <string>
#include <vector>

// Convenience constants
constexpr double PI = std::numbers::pi;
using Clock = std::chrono::high_resolution_clock;

double f(double x, double y) { return -8 * PI * PI * sin(2 * PI * x) * cos(2 * PI * y); }

int main(int argc, char* argv[]) {
    // Verify OpenMP is available
#ifdef _OPENMP
    std::cout << "OpenMP is enabled with " << omp_get_max_threads() << " threads." << std::endl;
#else
    std::cerr << "Error: OpenMP is not enabled. Please compile with OpenMP support." << std::endl;
    return 1;
#endif

    // Parse command line arguments for verbosity and grid size
    bool verbose = false;
    int N = 250;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            try {
                N = std::stoi(arg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid argument: " << arg << ". Expected an integer for grid size." << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Using grid size N = " << N << std::endl;

    // Define constants
    const double start = 0.0;
    const double end = 1.0;
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-3;

    // Initialize the grid, new grid, and f values
    std::vector<double> u(N * N, 0.0);
    std::vector<double> u_new(N * N, 0.0);
    std::vector<double> f_values(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = start + j * h;
            double y = start + i * h;
            f_values[i * N + j] = f(x, y);
        }
    }

    // Begin timing
    auto start_time = Clock::now();

    // Jacobi iteration
    int iterations = 0;
    while (true) {
        // Update internal grid points
#pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double f_term = f_values[i * N + j] * h * h * -1;
                double neighbor_term =
                    u[(i - 1) * N + j] + u[(i + 1) * N + j] + u[i * N + (j - 1)] + u[i * N + (j + 1)];
                u_new[i * N + j] = 0.25 * (neighbor_term + f_term);
            }
        }

        // Swap grids
        std::swap(u, u_new);

        // Compute residual
        double max_residual = 0.0;
#pragma omp parallel for collapse(2) reduction(max : max_residual)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double x_partial = (u[(i - 1) * N + j] - 2 * u[i * N + j] + u[(i + 1) * N + j]) / (h * h);
                double y_partial = (u[i * N + (j - 1)] - 2 * u[i * N + j] + u[i * N + (j + 1)]) / (h * h);
                double gradient = x_partial + y_partial;
                double residual = std::abs(gradient - f_values[i * N + j]);
                max_residual = std::max(max_residual, residual);
            }
        }

        // Update metrics
        ++iterations;

        // Check for convergence
        if (max_residual < tolerance) {
            break;
        } 

        // Optional: Print progress every 100 iterations
        if (verbose && iterations % 1000 == 0) {
            std::cout << "Iteration " << iterations << ", Max Residual: " << max_residual << std::endl;
        }
    }

    // End timing
    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}