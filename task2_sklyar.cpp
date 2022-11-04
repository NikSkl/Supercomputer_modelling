#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <string>

using namespace std;

class Func {
private: 
    double x;
    double y;
    double z;
public:
    void X(double value) {
        x = value;
    }
    void Y(double value) {
        y = value;
    }
    void Z(double value) {
        z = value;
    }
    double function() {
        return ((fabs(x) + fabs(y)) <= 1) and (z >= -2) and (z <= 2) ? x * x * y * y * z * z : 0;
    }
};

int main(int argc, char* argv[])
{
    double eps = atof(argv[1]);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const double exact_solution = 16.0 / 135, upper_volume = 16.0;
    double error = 1.0;
    int sub_dots_counter = 3;
    double sum = 0.0;
    double max_time = 0.0;
    double j_sum = 0.0;
    long int dots_count = sub_dots_counter * (size - 1);
    double* dots = new double[3 * dots_count];
    double* sub_dots = new double[3 * sub_dots_counter];
    long long int k = 0;
    int working = 1;
    double start_time = MPI_Wtime();
    double root_time = 0.0;

    if (rank == 0) {
        while (error > eps) {
            double root_time_start = MPI_Wtime();
            for (int j = 0; j < 3 * dots_count; j += 3) {
                dots[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                dots[j + 1] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                dots[j + 2] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
            }

            for (int j = 0; j < size - 1; j++) {
                MPI_Send(&dots[j * 3 * sub_dots_counter], 3 * sub_dots_counter, MPI_DOUBLE, j + 1, 0, MPI_COMM_WORLD);
            }

            MPI_Reduce(&sum, &j_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            k++;
            error = abs(upper_volume * j_sum / (dots_count * k) - exact_solution);

            working = error > eps ? 1 : 0;
            MPI_Bcast(&working, 1, MPI_INT, 0, MPI_COMM_WORLD);
            double root_time_finish = MPI_Wtime();
            root_time += root_time_finish - root_time_start;
        }
    }
    else {
        while (working == 1) {
            MPI_Recv(sub_dots, 3 * sub_dots_counter, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < 3 * sub_dots_counter; i += 3) {
                Func func;
                func.X(sub_dots[i]);
                func.Y(sub_dots[i + 1]);
                func.Z(sub_dots[i + 2]);
                sum += func.function();
            }

            MPI_Reduce(&sum, &j_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&working, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    double time = MPI_Wtime() - start_time - root_time;

    if (rank == 0) {
        cout << "Integral =  " << upper_volume * j_sum / (dots_count * k) << endl;
        cout << "err = " << error << endl;
        cout << "N = " << dots_count * k << endl;
        cout << "time = " << time << endl;
    }

    MPI_Finalize();
    delete[] dots;
    delete[] sub_dots;
    return 0;
}
