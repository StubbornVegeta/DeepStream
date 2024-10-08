///////////////////////////////////////////////////////////////////////////
//
// PicoPebble - A lightweight distributed machine learning training framework for beginners
//
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024. All rights reserved.
//
// Licensed under the MIT License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////////

/**
 * Template function from Trainer class implementation
 */

#include "Common.hpp"
#include <chrono>
#include <cmath>
#include <thread>

using namespace DeepLearningFramework;

inline Eigen::MatrixXf oneHotEncoding(const Eigen::MatrixXf &input) {
  /* Transform labels into one-hot encoding:
   * 0  ->  0 0 0
   * 1  ->  0 1 0
   * 2  ->  0 0 1
   * */
  int n = input.rows();
  int N_classes = input.maxCoeff() + 1;
  Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(n, N_classes);
  for (int i = 0; i < n; ++i) {
    one_hot(i, static_cast<int>(input(i))) = 1.0;
  }
  return one_hot;
}

template <uint32_t batch_size, uint32_t feature_dim>
void Trainer::trainModel(std::vector<float> train_acc,
                         std::vector<float> test_acc, Sequential &model,
                         uint32_t epochs, const Eigen::MatrixXf &y_train,
                         const Eigen::MatrixXf &X_train,
                         const Eigen::MatrixXf &y_test,
                         const Eigen::MatrixXf &X_test, uint32_t step) {

  uint32_t batch_num = X_train.rows() / batch_size;

  // set the flag for stopping synchronization parameters.
  trainFinishFlag().setStatus(epochs - 1, batch_num - 1);

  for (uint32_t i = 0; i < epochs; i++) {
    float loss = 0.f;
    for (uint32_t batch_idx = 0; batch_idx < batch_num; batch_idx++) {
      float batch_loss = 0.f;
      globalTrainStatus().setStatus(i, batch_idx);
      Eigen::MatrixXf X_batch =
          X_train.block<batch_size, feature_dim>(batch_idx * batch_size, 0);
      Eigen::MatrixXf y_batch =
          y_train.block<batch_size, 1>(batch_idx * batch_size, 0);
      y_batch = oneHotEncoding(y_batch);

      model.forward(X_batch);
      model.backward(batch_loss, y_batch, X_batch);
      loss += batch_loss;
    }

    addAccuracy(train_acc, model, y_train, X_train);
    addAccuracy(test_acc, model, y_test, X_test);

    loss /= batch_num;

    if (i % step == 0)
      Log() << "Epoch: " << i << ", train accuracy: " << train_acc.at(i)
            << ", loss: " << loss << ", test accuracy: " << test_acc.at(i);
  }
}
