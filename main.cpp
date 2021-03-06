#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>

#include "mpi.h"

using namespace std;

#define dataType double
MPI_Datatype MPI_DATATYPE = MPI_DOUBLE;

#define PRESUMED_RANK (workData.GetCols()-1)%procAmount
#define CUR_PRESUMED_RANK int(i%procAmount)

class Matrix
{
    size_t rows;
    size_t cols;
    vector<dataType> matrix;

public:
    Matrix(size_t rows, size_t cols, dataType *mtr = NULL) : rows(rows), cols(cols), matrix(vector<dataType>(rows * cols, 0))
    {
        if (mtr)
            matrix.assign(size_t(mtr), matrix.size());
    }

    Matrix(const Matrix &mtr) { rows = mtr.rows; cols = mtr.cols; matrix = mtr.matrix; }

    dataType *GetCol(size_t col) { return matrix.data() + col * rows; }

    size_t GetRows() { return rows; }

    dataType& operator()(size_t i, size_t j) { return matrix[j * rows + i]; }

    void Print()
    {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = i; j < matrix.size(); j += rows)
                cout << scientific << matrix[j] << ' ';
            cout << endl;
        }
        cout << endl;
    }
};

class WorkData
{
    int procRank, procAmount, offset;
    size_t rows, cols, localCols;
    ifstream in;

    WorkData(const WorkData&);
    WorkData& operator=(const WorkData&);

    dataType f(size_t i, size_t j)
    {
        //return 1 / dataType(i + j + 1);
        return max(i, j);
    }

    dataType f(ifstream &in)
    {
        dataType inData;
        in >> inData;
        return inData;
    }

public:
    WorkData(int procRank, int procAmount, const char *info) :
            procRank(procRank), procAmount(procAmount)
    {
        in.open(info);
        if (in.is_open())
            in >> rows >> cols;
        else {
            rows = size_t(atoi(info));
            cols = rows;
        }

        if (cols != rows) {
            cout << "Bad matrix params!" << endl;
            exit(1);
        }

        cols = rows + 1;

        localCols = cols / procAmount;
        if (cols % procAmount != 0 && procRank < cols % procAmount)
            localCols++;
    }

    void RemoveRightColumn() { if ((cols - 1) % procAmount == procRank) localCols--; }

    size_t GetLocalCols() { return localCols; }

    size_t GetCols() { return cols; }

    Matrix MakeMatrix()
    {
        Matrix matrix(rows, localCols);

        for (size_t i = 0; i < rows; i++) {
            offset = procRank;
            for (size_t j = 0; j < cols; j++) {
                dataType tmp = in.is_open() ? f(in) : f(i, j);
                if (j == rows && !in.is_open()) {
                    dataType summ = 0;
                    for (size_t k = 0; k < rows; k++) {
                        int presumedRank = int(k % procAmount);
                        if (presumedRank == procRank)
                            summ += matrix(i, k / procAmount);
                    }
                    MPI_Reduce(&summ, &tmp, 1, MPI_DATATYPE, MPI_SUM, int((cols - 1) % procAmount), MPI_COMM_WORLD);
                }
                if (j % procAmount == procRank) {
                    matrix(i, j - offset) = tmp;
                    offset += procAmount - 1;
                }
            }
        }

        in.close();

        return matrix;
    }
};

dataType ScalarProduct(const dataType *vec1, const dataType *vec2, size_t size)
{
    dataType sum = 0;
    for (size_t i = 0; i < size; i++)
        sum += vec1[i] * vec2[i];
    return sum;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int procRank, procAmount;
    MPI_Comm_size(MPI_COMM_WORLD, &procAmount);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (argc == 1) {
        cout << "Input matrix file or matrix size!" << endl;
        exit(1);
    }
    WorkData workData(procRank, procAmount, argv[1]);
    Matrix matrix(workData.MakeMatrix());
    Matrix matrixBackup(matrix);

    //matrix.Print();

    MPI_Barrier(MPI_COMM_WORLD);
    double startForward = MPI_Wtime();

    vector<dataType> solution(matrix.GetRows());
    int offset = procRank;
    for (size_t i = 0; i < solution.size(); i++) {
        if (CUR_PRESUMED_RANK == procRank) {
            dataType *curCol = matrix.GetCol(i - offset);
            solution = vector<dataType>(solution.size(), 0);
            copy(curCol + i, curCol + solution.size(), solution.begin() + i);
            solution[i] -= sqrt(ScalarProduct(solution.data() + i, solution.data() + i, solution.size() - i));
            dataType norm = sqrt(ScalarProduct(solution.data(), solution.data(), solution.size()));

            if (norm > numeric_limits<dataType>::min())
                for (size_t j = 0; j < solution.size(); j++)
                    solution[j] /= norm;

            offset += procAmount - 1;
        }

        MPI_Bcast(solution.data(), int(solution.size()), MPI_DATATYPE, CUR_PRESUMED_RANK, MPI_COMM_WORLD);

        for (size_t curColIndex = i / procAmount; curColIndex < workData.GetLocalCols(); curColIndex++) {
            dataType *curCol = matrix.GetCol(curColIndex);
            dataType scal = ScalarProduct(solution.data(), curCol, solution.size());
            for (size_t j = 0; j < solution.size(); j++)
                curCol[j] -= 2 * scal * solution[j];
        }

        //matrix.Print();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double endForward = MPI_Wtime();
    double startBackward = MPI_Wtime();

    workData.RemoveRightColumn();
    vector<dataType> rightColumn(solution.size());
    if (PRESUMED_RANK == procRank)
        copy(matrix.GetCol(workData.GetLocalCols()),
             matrix.GetCol(workData.GetLocalCols()) + rightColumn.size(), rightColumn.begin());
    MPI_Bcast(rightColumn.data(), int(rightColumn.size()), MPI_DATATYPE, int(PRESUMED_RANK), MPI_COMM_WORLD);

    solution = vector<dataType>(solution.size(), 0);
    size_t startCol = workData.GetLocalCols();
    for (int i = int(solution.size() - 1); i >= 0; i--) {
        dataType diff = 0, receiveBuffer;
        for (size_t j = startCol; j < workData.GetLocalCols(); j++)
            diff += matrix(size_t(i), j) * solution[j * procAmount + procRank];
        MPI_Reduce(&diff, &receiveBuffer, 1, MPI_DATATYPE, MPI_SUM, CUR_PRESUMED_RANK, MPI_COMM_WORLD);
        diff = receiveBuffer;

        if (CUR_PRESUMED_RANK == procRank) {
            startCol -= 1;
            solution[i] = (rightColumn[i] - diff) / matrix(size_t(i), size_t(i / procAmount));
        }
    }

    dataType *receiveBuffer = new dataType[solution.size()];
    MPI_Allreduce(solution.data(), receiveBuffer, int(solution.size()), MPI_DATATYPE, MPI_SUM, MPI_COMM_WORLD);
    copy(receiveBuffer, receiveBuffer + solution.size(), solution.begin());
    delete[] receiveBuffer;

    MPI_Barrier(MPI_COMM_WORLD);
    double endBackward = MPI_Wtime();

    vector<dataType> result(solution.size(), 0);
    for (size_t i = 0; i < result.size(); i++) {
        offset = procRank;
        for (size_t j = 0; j < workData.GetLocalCols(); j++) {
            result[i] += matrixBackup(i, j) * solution[offset];
            offset += procAmount;
        }
    }

    receiveBuffer = new dataType[solution.size()];
    MPI_Reduce(result.data(), receiveBuffer, int(result.size()), MPI_DATATYPE, MPI_SUM, int(PRESUMED_RANK), MPI_COMM_WORLD);
    copy(receiveBuffer, receiveBuffer + result.size(), result.begin());
    delete[] receiveBuffer;

    if (PRESUMED_RANK == procRank) {
        for (size_t i = 0; i < result.size(); i++)
            result[i] -= matrixBackup.GetCol(workData.GetLocalCols())[i];
        dataType discrepancy = sqrt(ScalarProduct(result.data(), result.data(), result.size()));

        for (size_t i = 0; i < solution.size(); i++)
            cout << solution[i] << ' ';
        cout << endl;
        cout <<   "Matrix size: " << matrix.GetRows()
             << "; Process amount: " << procAmount
             << "; Forward (microseconds): " << (endForward - startForward) * 1000000
             << "; Backward (microseconds): " << (endBackward - startBackward) * 1000000
             << "; Discrepancy: " << discrepancy << endl
             << endl;
    }

    MPI_Finalize();
    return 0;
}
