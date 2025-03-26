#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_ML_OPERATOR_CONVERTER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_ML_OPERATOR_CONVERTER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"

namespace blink::webnn_optimizer {

Node* ConvertMLOperatorToNode(const MLOperator* op);

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_ML_OPERATOR_CONVERTER_H_