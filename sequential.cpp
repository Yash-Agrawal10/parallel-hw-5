#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

// Convenience constants
constexpr double PI = std::numbers::pi;
using Clock = std::chrono::high_resolution_clock;

double f(double x, double y) { return -8 * PI * PI * sin(2 * PI * x) * cos(2 * PI * y); }

int main(int argc, char* argv[]) {
    // Parse command line arguments for verbosity
    bool verbose = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
    }

    // Define constants
    const double start = 0.0;
    const double end = 1.0;
    const int N = 100; // Number of grid points in each dimension
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-6;
    const int max_iterations = 100000;

    // Initialize the grid
    std::vector<std::vector<double>> u(N, std::vector<double>(N, 0.0));

    // Begin timing
    auto start_time = Clock::now();

    // Jacobi iteration
    int iterations = 0;
    while (true) {
        // Initialize new grid
        std::vector<std::vector<double>> u_new(N, std::vector<double>(N, 0.0));

        // Update internal grid points
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double x = start + i * h;
                double y = start + j * h;
                double f_term = f(x, y) * h * h * -1;
                double neighbor_term = u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1];
                u_new[i][j] = 0.25 * (neighbor_term + f_term);
            }
        }

        // Swap grids
        u = std::move(u_new);

        // Compute residual
        double max_residual = 0.0;
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double x = start + i * h;
                double y = start + j * h;
                double x_partial = (u[i - 1][j] - 2 * u[i][j] + u[i + 1][j]) / (h * h);
                double y_partial = (u[i][j - 1] - 2 * u[i][j] + u[i][j + 1]) / (h * h);
                double gradient = x_partial + y_partial;
                double residual = std::abs(gradient - f(x, y));
                max_residual = std::max(max_residual, residual);
            }
        }

        // Update metrics
        ++iterations;

        // Check for convergence
        if (max_residual < tolerance) {
            break;
        } else if (iterations >= max_iterations) {
            std::cout << "Reached maximum iterations without convergence." << std::endl;
            return 1;
        }

        // Optional: Print progress every 100 iterations
        if (verbose && iterations % 100 == 0) {
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