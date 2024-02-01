#pragma once

#include "mpiController.hpp"

namespace DeepLearningFramework {
inline static MPIController &globalController() {
  static MPIController mpi_controller;
  return mpi_controller;
}
} // namespace DeepLearningFramework
