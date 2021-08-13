// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Pipeline.h"

int main() {
  std::string modelDir = "./model.nb";
  std::string savedImagePath = "./";
  int cpuThreadNum = 1;
  std::string cpuPowerMode = "LITE_POWER_NO_BIND";
  int inputWidth = 640;
  int inputHeight = 640;
  const std::vector<float> inputMean = {0.485f, 0.456f, 0.406f};
  const std::vector<float> inputStd = {0.229f, 0.224f, 0.225f};
  auto ppl = new Pipeline(modelDir, cpuThreadNum, cpuPowerMode, inputWidth,
                          inputHeight, inputMean, inputStd);
  cv::Mat rgbaImage;
  rgbaImage = cv::imread("./test.jpg");
  cvtColor(rgbaImage, rgbaImage, cv::COLOR_BGR2BGRA);
  bool modified = ppl->Process(rgbaImage, savedImagePath);
}
