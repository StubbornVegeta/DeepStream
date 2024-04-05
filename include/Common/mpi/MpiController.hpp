#pragma once

#include "mpi/TypeTraits.hpp"
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

namespace DeepLearningFramework {

enum TAG {
  FORWARD_FLAG,
  FORWARD_SHAPE,
  FORWARD_PARAMETERS,
  BACKWARD_FLAG,
  BACKWARD_SHAPE,
  BACKWARD_PARAMETERS
};

inline bool &isSyncStopped() {
  static bool is_sync_stopped = false;
  return is_sync_stopped;
}

struct TrainStatus {
  int epoch;
  int batch_idx;

  TrainStatus() : epoch(0), batch_idx(0) {}

  TrainStatus(int _epoch, int _batch_idx)
      : epoch(_epoch), batch_idx(_batch_idx) {}

  TrainStatus(const TrainStatus &other) {
    this->epoch = other.epoch;
    this->batch_idx = other.batch_idx;
  }

  void setStatus(int _epoch, int _batch_idx) {
    epoch = _epoch;
    batch_idx = _batch_idx;
  }

  bool operator==(const TrainStatus &other) const {
    return this->epoch == other.epoch && this->batch_idx == other.batch_idx;
  }
};

inline TrainStatus &globalTrainStatus() {
  static TrainStatus train_status;
  return train_status;
}

inline TrainStatus &trainFinishFlag() {
  static TrainStatus train_status(-1, -1);
  return train_status;
}

class MPIController {
public:
  MPIController() : mpi_rank(-1), mpi_size(-1), mpi_comm(MPI_COMM_WORLD) {
    mpiInit();
  };

  ~MPIController() { mpiFinalize(); };

  int &mpiRank() { return mpi_rank; }

  int &mpiSize() { return mpi_size; };

  void mpiInit() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
  };

  void setGlobalDoneRankNum(const int &num) { _global_done_rank_num = num; }

  int getGlobalDoneRankNum() { return _global_done_rank_num; }

  template <typename T>
  int mpiScatterv(const std::vector<T> &sendbuf, const int counts[],
                  std::vector<T> &recvbuf, int recvcount, int root) {

    std::vector<int> displs(mpi_size);
    if (mpi_rank == root) {
      displs[0] = 0;
      for (int i = 1; i < mpi_size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
      }
    }

    return MPI_Scatterv(sendbuf.data(), counts, displs.data(),
                        getMPIDataType<T>(), recvbuf.data(), recvcount,
                        getMPIDataType<T>(), root, mpi_comm);
  }

  template <typename T>
  void mpiScatter(const T send_data[], int send_count, T &recv_data,
                  int recv_count, int root) {
    MPI_Scatter(send_data, send_count, getMPIDataType<T>(), &recv_data,
                recv_count, getMPIDataType<T>(), root, mpi_comm);
  }

  template <typename T> void mpiPull(T *sendbuf, T *recvbuf, int count) {
    MPI_Win win;
    MPI_Win_create(sendbuf, count * sizeof(T), sizeof(T), MPI_INFO_NULL,
                   mpi_comm, &win);
    if (mpi_rank != 0) {
      MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);

      MPI_Get(recvbuf, count, getMPIDataType<T>(), 0, 0, count,
              getMPIDataType<T>(), win);

      MPI_Win_unlock(0, win);
    }
    MPI_Win_free(&win);
  }

  // deprecated
  template <typename T>
  void mpiSync(T *sendbuf, T *recvbuf, int count, int tag) {
    MPI_Status mpi_status;
    if (mpi_rank == 0) {
      MPI_Request mpi_request[mpi_size - 1];
      for (int i = 1; i < mpi_size; i++) {
        MPI_Isend(sendbuf, count, getMPIDataType<T>(), i, tag, mpi_comm,
                  &mpi_request[i - 1]);
      }

      MPI_Waitall(mpi_size - 1, mpi_request, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(recvbuf, count, getMPIDataType<T>(), 0, tag, mpi_comm,
               &mpi_status);
    }
  }

  // template<typename T>
  // void mpiPull(T* sendbuf, T* recvbuf, int count, int tag) {
  //   if (mpi_rank == 0) {
  //     for(int i = 1; i < mpi_size; i++) {
  //         int request;
  //         MPI_Status mpi_status;
  //         MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
  //         MPI_COMM_WORLD, &mpi_status);

  //         MPI_Send(sendbuf, count, getMPIDataType<T>(), request,
  //         mpi_status.MPI_TAG, MPI_COMM_WORLD);
  //       }
  //   } else {
  //       // 非rank0节点发送数据请求
  //       MPI_Send(&mpi_rank, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
  //       // 接收来自rank0的数据
  //       MPI_Recv(recvbuf, count, getMPIDataType<T>(), 0, tag, MPI_COMM_WORLD,
  //       MPI_STATUS_IGNORE);
  //   }
  // }

  template <typename T>
  void mpiBcast(std::vector<T> &sendbuf, int count, int root) {
    MPI_Bcast(sendbuf.data(), count, getMPIDataType<T>(), root, mpi_comm);
    MPI_Barrier(mpi_comm);
  }

  template <typename T> void mpiBcast(T &sendbuf, int root) {
    MPI_Bcast(&sendbuf, 1, getMPIDataType<T>(), root, mpi_comm);
    MPI_Barrier(mpi_comm);
  }

  void mpiBarrier() { MPI_Barrier(mpi_comm); }

  template <typename T>
  void mpiPush(T *sendbuf, T *recvbuf, int count, int tag) {
    if (mpi_rank == 0) {
      int cnt = 0;
      T tmp_recvbuf[count];

      while (cnt < mpi_size - 1) {
        MPI_Recv(tmp_recvbuf, count, getMPIDataType<T>(), MPI_ANY_SOURCE, tag,
                 mpi_comm, MPI_STATUS_IGNORE);

        cnt++;
        for (int i = 0; i < count; i++) {
          recvbuf[i] += tmp_recvbuf[i];
        }
      }
    } else {
      MPI_Send(sendbuf, count, getMPIDataType<T>(), 0, tag, mpi_comm);
    }
  }

  void mpiForwardSend(int &forward_flag) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Send(&forward_flag, 1, MPI_INT, mpi_rank + 1, FORWARD_FLAG, mpi_comm);
    }
  }

  void mpiForwardRecv(int &forward_flag) {
    if (mpi_rank > 0) {
      MPI_Recv(&forward_flag, 1, MPI_INT, mpi_rank - 1, FORWARD_FLAG, mpi_comm,
               MPI_STATUS_IGNORE);
    }
  }

  void mpiForwardSend(std::vector<int> &shape) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Send(shape.data(), 2, MPI_INT, mpi_rank + 1, FORWARD_SHAPE, mpi_comm);
    }
  }

  void mpiForwardRecv(std::vector<int> &shape) {
    if (mpi_rank > 0) {
      MPI_Recv(shape.data(), 2, MPI_INT, mpi_rank - 1, FORWARD_SHAPE, mpi_comm,
               MPI_STATUS_IGNORE);
    }
  }

  template <typename T> void mpiForwardSend(T *array, int count) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Send(array, count, getMPIDataType<T>(), mpi_rank + 1,
               FORWARD_PARAMETERS, mpi_comm);
    }
  }

  template <typename T> void mpiForwardRecv(T *array, int count) {
    if (mpi_rank > 0) {
      MPI_Recv(array, count, getMPIDataType<T>(), mpi_rank - 1,
               FORWARD_PARAMETERS, mpi_comm, MPI_STATUS_IGNORE);
    }
  }

  void mpiBackwardSend(int &backward_flag) {
    if (mpi_rank > 0) {
      MPI_Send(&backward_flag, 1, MPI_INT, mpi_rank - 1, BACKWARD_FLAG,
               mpi_comm);
    }
  }

  void mpiBackwardRecv(int &backward_flag) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Recv(&backward_flag, 1, MPI_INT, mpi_rank + 1, BACKWARD_FLAG,
               mpi_comm, MPI_STATUS_IGNORE);
    }
  }

  void mpiBackwardSend(std::vector<int> &shape) {
    if (mpi_rank > 0) {
      MPI_Send(shape.data(), 2, MPI_INT, mpi_rank - 1, BACKWARD_SHAPE,
               mpi_comm);
    }
  }

  void mpiBackwardRecv(std::vector<int> &shape) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Recv(shape.data(), 2, MPI_INT, mpi_rank + 1, BACKWARD_SHAPE, mpi_comm,
               MPI_STATUS_IGNORE);
    }
  }

  template <typename T> void mpiBackwardSend(T *array, int count) {
    if (mpi_rank > 0) {
      MPI_Send(array, count, getMPIDataType<T>(), mpi_rank - 1,
               BACKWARD_PARAMETERS, mpi_comm);
    }
  }

  template <typename T> void mpiBackwardRecv(T *array, int count) {
    if (mpi_rank < mpi_size - 1) {
      MPI_Recv(array, count, getMPIDataType<T>(), mpi_rank + 1,
               BACKWARD_PARAMETERS, mpi_comm, MPI_STATUS_IGNORE);
    }
  }

  template <typename T>
  void mpiReduce(T *sendbuf, T *recvbuf, int count, const MPI_Op &op,
                 int root) {
    MPI_Reduce(sendbuf, recvbuf, count, getMPIDataType<T>(), op, root,
               mpi_comm);
  }

  void mpiFinalize() { MPI_Finalize(); };

private:
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm;
  int _global_done_rank_num;
};

} // namespace DeepLearningFramework
