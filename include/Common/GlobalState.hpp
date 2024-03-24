#pragma once

#include "GlobalState.hpp"
#include "mpi/MpiController.hpp"
#include <Eigen/Dense>
#include <Module.hpp>
#include <iostream>
#include <vector>

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>

namespace DeepLearningFramework {

enum ParallelismMode {
  DATA_PARALLELISM,
  TENSOR_MODEL_PARALLELISM,  // Inter-Layer Model Parallelism
  PIPELINE_MODEL_PARALLELISM // Intra-Layer Model Parallelism
};

enum TrainMode { SYNC, ASYNC };

template <typename F, typename... Args> struct invoke_result {
  using type = decltype(std::declval<F>()(std::declval<TrainStatus>(),
                                          std::declval<Args>()...));
};

template <typename F, typename... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

class BackgroundThread {
public:
  explicit BackgroundThread() : m_stop(false) {
    worker = std::thread([this]() {
      while (true) {
        std::unique_lock<std::mutex> lock(this->m_mtx);

        this->m_cv.wait(
            lock, [this]() { return this->m_stop || !this->tasks.empty(); });

        if (this->m_stop && this->tasks.empty())
          return;

        std::function<void()> task = std::move(this->tasks.front());
        this->tasks.pop();

        lock.unlock();

        task();
      }
    });
  }

  template <typename F, typename... Args>
  auto enqueue(F &&f, TrainStatus train_status, Args &&...args)
      -> std::future<invoke_result_t<F, Args...>> {
    using return_type = invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(
        std::forward<F>(f), train_status, std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(m_mtx);

      if (m_stop) {
        throw std::runtime_error("enqueue on stopped Thread pool");
      }

      // add task
      tasks.emplace([task = std::move(task),
                     train_status = std::move(train_status)]() { (*task)(); });
    }
    m_cv.notify_one();
    return res;
  }

  ~BackgroundThread() {
    {
      std::unique_lock<std::mutex> lock(m_mtx);
      m_stop = true;
    }
    m_cv.notify_all();

    worker.join();
  }

private:
  std::thread worker;
  std::queue<std::function<void()>> tasks;
  std::mutex m_mtx;
  std::condition_variable m_cv;
  bool m_stop;
};

inline BackgroundThread &globalBackgroundThread() {
  static BackgroundThread bgthread;
  return bgthread;
}

inline MPIController &globalController() {
  static MPIController mpi_controller;
  return mpi_controller;
}

inline TrainMode &globalTrainMode() {
  static TrainMode global_train_mode = SYNC;
  return global_train_mode;
}

inline ParallelismMode &globalParallelismMode() {
  static ParallelismMode global_parallelism_mode = TENSOR_MODEL_PARALLELISM;
  return global_parallelism_mode;
}

class GlobalState {
public:
  GlobalState() { initParameterInterval(); }
  void setLayersSize(const std::vector<int> &layers_size) {
    _layers_size = layers_size;
  }

  std::vector<int> getLayersSize() { return _layers_size; }

  int getLayersNum() { return _layers_size.size(); }

  std::vector<int> getNodeLayersRank() {
    return _node_layers_rank[globalController().mpiRank()];
  }

  std::vector<int> getLayerRankBound() {
    auto minmax = std::minmax_element(
        _node_layers_rank[globalController().mpiRank()].begin(),
        _node_layers_rank[globalController().mpiRank()].end());

    int min_layer_rank = *minmax.first;
    int max_layer_rank = *minmax.second;
    return {min_layer_rank, max_layer_rank};
  }

  void layersDistribution() {
    if (globalParallelismMode() == PIPELINE_MODEL_PARALLELISM) {
      const int mpi_size = globalController().mpiSize();
      const int layers_num = _layers_size.size();
      int node_layers_num[mpi_size];

      for (int i = 0; i < mpi_size; i++) {
        node_layers_num[i] = (layers_num - 1) / mpi_size;
      }

      int layers_remainder = (layers_num - 1) % mpi_size;

      for (int i = 0; i < layers_remainder; i++) {
        node_layers_num[i]++;
      }

      int layer_indicator = 1;
      for (int i = 0; i < mpi_size; i++) {
        std::vector<int> layers_size, layers_rank;
        while (node_layers_num[i]--) {
          layers_size.emplace_back(_layers_size[layer_indicator - 1]);
          layers_size.emplace_back(_layers_size[layer_indicator]);
          layers_rank.emplace_back(layer_indicator - 1);
          layer_indicator++;
        }
        _node_layers_size.emplace_back(layers_size);
        _node_layers_rank.emplace_back(layers_rank);
      }

    } else {
      const int mpi_size = globalController().mpiSize();
      for (int i = 0; i < mpi_size; i++) {
        _node_layers_size.emplace_back(_layers_size);
      }
    }
  }

  void initParameterInterval() {
    switch (globalParallelismMode()) {
    case DATA_PARALLELISM: {
      _interval = 1;
      break;
    }
    case PIPELINE_MODEL_PARALLELISM: {
      _interval = 2;
      break;
    }
    case TENSOR_MODEL_PARALLELISM: {
    }
    }
  }

  void initGlobalWeights() {
    const int mpi_rank = globalController().mpiRank();
    for (int i = 1; i < _node_layers_size[mpi_rank].size(); i = i + _interval) {
      _global_weigths.emplace_back(Eigen::MatrixXf::Random(
          _node_layers_size[mpi_rank][i - 1], _node_layers_size[mpi_rank][i]));
    }
  }

  void initGlobalBias() {
    const int mpi_rank = globalController().mpiRank();
    for (int i = 1; i < _node_layers_size[mpi_rank].size(); i = i + _interval) {
      _global_bias.emplace_back(
          Eigen::MatrixXf::Random(1, _node_layers_size[mpi_rank][i]));
    }
  }

  void setWeights(Eigen::MatrixXf &weights, const int &layer_rank) {
    _global_weigths[layer_rank] = weights;
  }

  void setBias(Eigen::MatrixXf &bias, const int &layer_rank) {
    _global_bias[layer_rank] = bias;
  }

  void setGlobalBias() {
    const int layers_num = _layers_size.size();
    for (int i = 1; i < layers_num; ++i) {
      _global_bias.emplace_back(Eigen::MatrixXf::Random(1, _layers_size[i]));
    }
  }

  Eigen::MatrixXf &getWeights(const int &layer_rank) {
    return _global_weigths[layer_rank];
  }

  Eigen::MatrixXf &getBias(const int &layer_rank) {
    return _global_bias[layer_rank];
  }

private:
  std::vector<Eigen::MatrixXf> _global_weigths;
  std::vector<Eigen::MatrixXf> _global_bias;
  std::vector<int> _layers_size;
  std::vector<std::vector<int>> _node_layers_size;
  std::vector<std::vector<int>> _node_layers_rank;
  int _interval;
};

inline GlobalState &globalState() {
  static GlobalState global_state;
  return global_state;
}

inline int &minLayerRank() {
  static int min_layer_rank = globalState().getLayerRankBound()[0];
  return min_layer_rank;
}

inline int &maxLayerRank() {
  static int max_layer_rank = globalState().getLayerRankBound()[1];
  return max_layer_rank;
}

inline std::vector<Module *> &globalModel() {
  static std::vector<Module *> global_model;
  return global_model;
}
inline Eigen::MatrixXf &firstLayerInput() {
  static Eigen::MatrixXf first_layer_input;
  return first_layer_input;
}
} // namespace DeepLearningFramework
