#pragma once

#include "GlobalState.hpp"
#include "mpi/MpiController.hpp"
#include <Eigen/Dense>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace DeepLearningFramework {

class Log {
public:
  Log() { std::cout << "Rank: " << globalController().mpiRank() << ", "; }

  ~Log() { std::cout << std::endl; }

  template <typename T> Log &operator<<(const T &message) {
    std::cout << message;
    return *this;
  }
};

inline int &globalDoneRankNum() {
  static int done_rank_num = 0;
  return done_rank_num;
}

template <typename... Args> class Decorator {
public:
  Decorator(std::function<void(Args &&...)> f) : m_func(f) {}
  void operator()(Args &&...args) { m_func(std::forward<Args>(args)...); }

protected:
  std::function<void(Args &&...)> m_func;
};

template <typename... Args>
class SyncStatusDecorator : public Decorator<Args &&...> {
public:
  SyncStatusDecorator(std::function<void(Args &&...)> f)
      : Decorator<Args &&...>(f) {}
  void operator()(TrainStatus train_status, Args &&...args) {

    MPIController &global_controller = globalController();

    if (globalDoneRankNum() == global_controller.mpiSize() - 1) {
      return;
    }

    Decorator<Args &&...>::operator()(std::forward<Args>(args)...);

    if (train_status == trainFinishFlag()) {
      done_status = 1;
    }

    global_controller.mpiReduce<int>(&done_status, &done_rank_num, 1, MPI_SUM,
                                     0);
    globalDoneRankNum() = done_rank_num;
  }

private:
  int done_status = 0;
  int done_rank_num = 0;
};

inline std::string formatString(int number) {
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << number;
  return ss.str();
}

inline void convertMatrixToArray(const Eigen::MatrixXf &mat, float *buf) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      buf[i * mat.cols() + j] = mat(i, j);
    }
  }
}

inline void convertArrayToMatrix(float *buf, Eigen::MatrixXf &mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      mat(i, j) = buf[i * mat.cols() + j];
    }
  }
}

inline void pullParameters() {
  GlobalState &global_state = globalState();
  MPIController &global_controller = globalController();

  const int layers_num = globalState().getLayersNum();
  for (int i = 0; i < layers_num - 1; i++) {
    // weight
    Eigen::MatrixXf &weights_mat = globalState().getWeights(i);
    int count = weights_mat.rows() * weights_mat.cols();
    float send_weigths_buf[count];
    float recv_weigths_buf[count];
    if (global_controller.mpiRank() == 0) {
      convertMatrixToArray(weights_mat, send_weigths_buf);
    }
    global_controller.mpiPull<float>(send_weigths_buf, recv_weigths_buf, count);
    // global_controller.mpiSync<float>(send_weigths_buf, recv_weigths_buf,
    // count, 0);

    if (global_controller.mpiRank() != 0) {
      convertArrayToMatrix(recv_weigths_buf, weights_mat);
    }

    // bias
    Eigen::MatrixXf &bias_mat = globalState().getBias(i);
    count = bias_mat.rows() * bias_mat.cols();
    float send_bias_buf[count];
    float recv_bias_buf[count];
    if (global_controller.mpiRank() == 0) {
      convertMatrixToArray(bias_mat, send_bias_buf);
    }

    global_controller.mpiPull<float>(send_bias_buf, recv_bias_buf, count);
    // global_controller.mpiSync<float>(send_bias_buf, recv_bias_buf, count, 0);

    if (global_controller.mpiRank() != 0) {
      convertArrayToMatrix(recv_bias_buf, bias_mat);
    }
  }
}

inline void pushGradients(Eigen::MatrixXf &grad, const int &tag) {
  MPIController &global_controller = globalController();

  int count = grad.rows() * grad.cols();
  float send_grad_buf[count];
  float recv_grad_buf[count];
  if (global_controller.mpiRank() == 0) {
    convertMatrixToArray(grad, recv_grad_buf);
  } else {
    convertMatrixToArray(grad, send_grad_buf);
  }

  global_controller.mpiPush<float>(send_grad_buf, recv_grad_buf, count, tag);

  if (global_controller.mpiRank() == 0) {
    convertArrayToMatrix(recv_grad_buf, grad);
  }
}

inline void barrier() {
  globalController().mpiBarrier();
  return;
}

static SyncStatusDecorator<> Barrier(barrier);

static SyncStatusDecorator<> PullParameters(pullParameters);

/* deprecated: parameters update interval */
/*
inline void PullParametersStep(int step) {
  if (step == 0) {
    return PullParameters();
  }
  static int cnt = -1;

  cnt++;
  if (cnt % step != 0) {
    return;
  } else {
    PullParameters();
  }
}
*/

static SyncStatusDecorator<Eigen::MatrixXf &, const int &>
    PushGradients(pushGradients);

inline void initialize(const std::vector<int> &layers_size) {
  MPIController &global_controller = globalController();

  GlobalState &global_state = globalState();
  global_state.setLayersSize(layers_size);

  if (globalParallelismMode() == PIPELINE_MODEL_PARALLELISM &&
      global_state.getLayersNum() - 1 < global_controller.mpiSize()) {
    Log() << "The network’s parameters can be distributed to a maximum of "
          << global_state.getLayersNum() - 1 << " nodes.";
    exit(1);
  }
  global_state.layersDistribution();

  /* init global weight and bias */
  global_state.initGlobalWeights();
  global_state.initGlobalBias();

  if (globalParallelismMode() == DATA_PARALLELISM) {
    PullParameters(globalTrainStatus());
  }
}

inline void finalize() {
  // Log() << "finalize";
}

} // namespace DeepLearningFramework
