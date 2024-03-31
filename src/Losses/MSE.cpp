/**
 * MSE loss class implementation
 */

#include "MSE.hpp"
#include "GlobalState.hpp"

#include <iostream>

using namespace DeepLearningFramework::Losses;

MSE::MSE() {}

void MSE::forward(float &loss, const Eigen::MatrixXf &y,
                  const Eigen::MatrixXf &y_pred) {
  loss = (y_pred - y).squaredNorm() / y.rows();
}

void MSE::backward(Eigen::MatrixXf &dloss, const Eigen::MatrixXf &y,
                   const Eigen::MatrixXf &y_pred) {
  // dloss = 2.f * (y_pred - y) / y.rows();
  switch (globalParallelismMode()) {
  case DATA_PARALLELISM: {
    dloss = 2.f * (y_pred - y) / y.rows();
    break;
  }
  case PIPELINE_MODEL_PARALLELISM: {
    if (microBatchLossFlag()) {
      dloss = 2.f * (y_pred - y) / y.rows();
      if (globalMicroBatchIdx() == globalMicroBatchNum() - 1) {
        _dloss = dloss;
      } else {
        _dloss += dloss;
      }
    } else if (globalMicroBatchNum() == 0) {
      dloss = 2.f * (y_pred - y) / y.rows();
    } else {
      dloss = _dloss;
      _dloss.setZero();
    }
    break;
  }

  case TENSOR_MODEL_PARALLELISM: {
    dloss = 2.f * (y_pred - y) / y.rows();
    break;
  }
  }
}

void MSE::printDescription() { std::cout << "MSE loss" << std::endl; }

std::string MSE::getName() { return _name; }
