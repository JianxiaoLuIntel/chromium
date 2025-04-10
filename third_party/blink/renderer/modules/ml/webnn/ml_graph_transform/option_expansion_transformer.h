#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_OPTION_EXPANSION_TRANSFORMER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_OPTION_EXPANSION_TRANSFORMER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/ml_graph_transformer.h"

namespace blink {

class OptionExpansionTransformer : public MLGraphTransformer {
 public:
  explicit OptionExpansionTransformer(MLGraphBuilder* graph_builder)
      : MLGraphTransformer(graph_builder) {}

  void Transform(MLNamedOperands& named_outputs) override;

 private:
  template <typename MLConv2dOptionsType>
  MLOperand* HandleConv2d(MLOperator* conv2d);
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_OPTION_EXPANSION_TRANSFORMER_H_
