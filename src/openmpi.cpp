#include <mpi.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// Convenience constants
constexpr double PI = std::numbers::pi;
using Clock = std::chrono::high_resolution_clock;

double f(double x, double y) { return -8 * PI * PI * sin(2 * PI * x) * cos(2 * PI * y); }

int main(int argc, char* argv[]) {
    // Set up MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments for verbosity and grid size
    bool verbose = false;
    int N = 256;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            try {
                N = std::stoi(arg);
            } catch (const std::invalid_argument&) {
                if (rank == 0) {
                    std::cerr << "Invalid argument: " << arg << ". Expected an integer for grid size." << std::endl;
                }
                MPI_Finalize();
                return 1;
            }
        }
    }
    if (rank == 0) {
        std::cout << "Using grid size N = " << N << std::endl;
    }

    // Define global constants
    const double start = 0.0;
    const double end = 1.0;
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-3;

    // Define local variables
    const int N_local = (rank < N % size) ? (N / size + 1) : (N / size);
    int N_previous = 0;
    for (int r = 0; r < rank; ++r) {
        N_previous += (r < N % size) ? (N / size + 1) : (N / size);
    }
    const double start_local = start + N_previous * h;
    const double end_local = start_local + (N_local - 1) * h;

    // Initialize the local grid and new grid
    std::vector<double> u(N_local * N, 0.0);
    std::vector<double> u_new(N_local * N, 0.0);
    std::vector<double> f_values(N_local * N, 0.0);
    for (int i = 0; i < N_local; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = start + j * h;
            double y = start_local + i * h;
            f_values[i * N + j] = f(x, y);
        }
    }

    // Initialize communication buffers
    std::vector<double> send_previous_i(N, 0.0);
    std::vector<double> recv_previous_i(N, 0.0);
    std::vector<double> send_next_i(N, 0.0);
    std::vector<double> recv_next_i(N, 0.0);

    // Begin timing
    auto start_time = Clock::now();

    // Jacobi iteration
    int iterations = 0;
    while (true) {
        // Do communication for boundary data
        MPI_Request requests[4];
        int req_count = 0;

        if (rank > 0) {
            for (int j = 0; j < N; ++j) {
                send_previous_i[j] = u[j];
            }
            MPI_Irecv(recv_previous_i.data(), N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Isend(send_next_i.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
        }

        if (rank < size - 1) {
            for (int j = 0; j < N; ++j) {
                send_next_i[j] = u[(N_local - 1) * N + j];
            }
            MPI_Irecv(recv_next_i.data(), N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
            MPI_Isend(send_previous_i.data(), N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
        }

        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Update internal grid points
        for (int i = 0; i < N_local; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                if ((i == 0 && rank == 0) || (i == N_local - 1 && rank == size - 1)) {
                    continue; // Skip global boundary points
                }
                double f_term = f_values[i * N + j] * h * h * -1;
                double top = (i == 0) ? recv_previous_i[j] : u[(i - 1) * N + j];
                double bottom = (i == N_local - 1) ? recv_next_i[j] : u[(i + 1) * N + j];
                double neighbor_term = top + bottom + u[i * N + (j - 1)] + u[i * N + (j + 1)];
                u_new[i * N + j] = 0.25 * (neighbor_term + f_term);
            }
        }

        // Swap grids
        std::swap(u, u_new);

        // Compute local residual
        double local_max_residual = 0.0;
        for (int i = 0; i < N_local; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                if ((i == 0 && rank == 0) || (i == N_local - 1 && rank == size - 1)) {
                    continue; // Skip global boundary points
                }
                double y_partial = ((i == 0) ? recv_previous_i[j] : u[(i - 1) * N + j]) - 2 * u[i * N + j] +
                                   ((i == N_local - 1) ? recv_next_i[j] : u[(i + 1) * N + j]);
                double x_partial = u[i * N + (j - 1)] - 2 * u[i * N + j] + u[i * N + (j + 1)];
                double gradient = x_partial + y_partial;
                double residual = std::abs(gradient - f_values[i * N + j]);
                local_max_residual = std::max(local_max_residual, residual);
            }
        }

        // Reduce to find global maximum residual
        double global_max_residual = 0.0;
        MPI_Allreduce(&local_max_residual, &global_max_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Update metrics
        ++iterations;

        // Check for convergence
        if (global_max_residual < tolerance) {
            break;
        }

        // Optional verbose output
        if (verbose && rank == 0 && iterations % 100 == 0) {
            std::cout << "Iteration " << iterations << ", max residual = " << global_max_residual << std::endl;
        }
    }

    // End timing
    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Output results
    if (rank == 0) {
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}