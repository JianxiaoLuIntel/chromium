// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_PRINTER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_PRINTER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
namespace blink {

void Print(MLNamedOperands& named_outputs);

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_PRINTER_H_
