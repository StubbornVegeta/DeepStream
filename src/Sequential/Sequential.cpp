/**
 * Sequential model class implementation
 */

#include "Sequential.hpp"
#include "Common.hpp"
#include "GlobalState.hpp"
#include <algorithm>

#include <iostream>
#include <mpi.h>

using namespace DeepLearningFramework;

Sequential::Sequential(std::vector<Module *> &model, Losses::MSE loss) {
  switch (globalParallelismMode()) {
  case DATA_PARALLELISM: {
    _model = model;
    break;
  }
  case PIPELINE_MODEL_PARALLELISM: {
    std::vector<int> node_layer_rank = globalState().getNodeLayersRank();
    int node_layer_num = node_layer_rank.size() * 2;
    int node_layer_start_rank = node_layer_rank[0] * 2;
    _model.resize(node_layer_num);
    std::copy(model.begin() + node_layer_start_rank,
              model.begin() + node_layer_start_rank + node_layer_num,
              _model.begin());
    globalModel() = _model;
    break;
  }
  case TENSOR_MODEL_PARALLELISM: {
  }
  }

  if (globalController().mpiRank() == 0) {
    _forward_flag = 1;
  } else {
    _forward_flag = 0;
  }

  if (globalController().mpiRank() == globalController().mpiSize() - 1) {
    _backward_flag = 0;
  } else {
    _backward_flag = 0;
  }

  _loss = loss;
}

void Sequential::forward(Eigen::MatrixXf &x, const std::string &mode) {
  std::vector<Module *>::iterator it;

  if (mode == "train" && globalParallelismMode() == DATA_PARALLELISM) {
    if (globalTrainMode() == SYNC) {
      PullParameters(globalTrainStatus());
    } else {
      globalBackgroundThread().enqueue(PullParameters, globalTrainStatus());
    }
  }

  for (int m = 0; m < _model.size(); m++) {
    if (globalParallelismMode() == DATA_PARALLELISM) {
      _model[m]->forward(x, x);
    } else {
      std::vector<int> x_shape(2);
      Eigen::MatrixXf last_layer_out;

      if (m == 0) {
        globalController().mpiForwardRecv(_forward_flag);
        if (globalController().mpiRank() != 0 && _forward_flag) {
          globalController().mpiForwardRecv(x_shape);
          last_layer_out.resize(x_shape[0], x_shape[1]);
          int count = x_shape[0] * x_shape[1];
          float last_layer_out_array[count];

          globalController().mpiForwardRecv(last_layer_out_array, count);
          convertArrayToMatrix(last_layer_out_array, last_layer_out);
        } else {
          last_layer_out = x;
        }
      } else {
        last_layer_out = x;
      }

      if (_forward_flag) {
        _model[m]->forward(x, last_layer_out);
      }
      if (m != _model.size() - 1) {
        continue;
      }
      x_shape[0] = x.rows();
      x_shape[1] = x.cols();

      float x_array[x.size()];
      convertMatrixToArray(x, x_array);
      globalController().mpiForwardSend(_forward_flag);

      globalController().mpiForwardSend(x_shape);

      globalController().mpiForwardSend(x_array, (int)x.size());
      _forward_flag = 0;
      if (mode != "train" && globalController().mpiRank() == 0) {
        _forward_flag = 1;
      }
    }
  }
  if (globalController().mpiRank() == globalController().mpiSize() - 1) {
    _backward_flag = 1;
  }
}

void Sequential::backward(float &loss, const Eigen::MatrixXf &y,
                          Eigen::MatrixXf &y_pred) {
  int tag = 0;
  Eigen::MatrixXf grad;
  if (globalParallelismMode() == DATA_PARALLELISM) {
    // calculate loss
    _loss.forward(loss, y, y_pred);

    // back propagation
    _loss.backward(grad, y, y_pred);
  } else {
    if (globalController().mpiRank() == globalController().mpiSize() - 1) {
      _loss.forward(loss, y, y_pred);
      _loss.backward(grad, y, y_pred);
    }
  }

  for (int m = _model.size() - 1; m > -1; m--) {
    if (globalParallelismMode() == DATA_PARALLELISM) {
      if (globalTrainMode() == SYNC) {
        PushGradients(globalTrainStatus(), grad, tag);
      } else {
        globalBackgroundThread().enqueue(PushGradients, globalTrainStatus(),
                                         grad, tag);
      }
      _model[m]->backward(grad, grad);
      tag++;
    } else {
      std::vector<int> grad_shape(2);
      Eigen::MatrixXf last_layer_out;
      if (m == _model.size() - 1) {
        globalController().mpiBackwardRecv(_backward_flag);
        if (globalController().mpiRank() != globalController().mpiSize() - 1) {
          globalController().mpiBackwardRecv(grad_shape);
          last_layer_out.resize(grad_shape[0], grad_shape[1]);
          int count = grad_shape[0] * grad_shape[1];
          float last_layer_out_array[count];
          globalController().mpiBackwardRecv(last_layer_out_array, count);
          convertArrayToMatrix(last_layer_out_array, last_layer_out);
        } else {
          last_layer_out = grad;
        }
      } else {
        last_layer_out = grad;
      }

      if (_backward_flag) {
        _model[m]->backward(grad, last_layer_out);
        tag++;
      }
      if (m > 0) {
        continue;
      }
      grad_shape[0] = grad.rows();
      grad_shape[1] = grad.cols();

      float grad_array[grad.size()];
      convertMatrixToArray(grad, grad_array);
      globalController().mpiBackwardSend(grad_shape);
      globalController().mpiBackwardSend(_backward_flag);
      globalController().mpiBackwardSend(grad_array, grad.size());
      _backward_flag = 0;
    }
  }

  if (globalController().mpiRank() == 0) {
    _forward_flag = 1;
  }
}

void Sequential::setLR(float lr) {
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    (*it)->setLR(lr);
}

uint32_t Sequential::getParametersCount() {
  uint32_t parametersCount = 0;
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    parametersCount += (*it)->getParametersCount();
  return parametersCount;
}

void Sequential::printDescription() {
  // layer description
  std::cout << "Model:" << std::endl;
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    (*it)->printDescription();

  // loss
  std::cout << "\nWith loss:" << std::endl;
  _loss.printDescription();

  // parameters count
  std::cout << "\nNumber of parameters:" << this->getParametersCount()
            << std::endl;
}

std::string Sequential::getName() { return _name; }
