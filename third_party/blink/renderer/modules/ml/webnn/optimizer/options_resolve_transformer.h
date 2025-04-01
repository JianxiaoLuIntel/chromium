#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OPTIMIZER_OPTIONS_RESOLVE_TRANSFORMER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OPTIMIZER_OPTIONS_RESOLVE_TRANSFORMER_H_

#include "third_party/blink/renderer/modules/ml/webnn/optimizer/graph_transformer.h"

namespace blink::webnn_optimizer {
// Used to resolve specific Operation options and ContextProperties which will
// alter graph structure.
class OptionsResolveTransformer : public GraphTransformer {
 public:
  OptionsResolveTransformer() = default;
  void TransformInternal() override;

 private:
 template< typename MLConv2dOptionsType>
  void ResolveConv2d(Conv2dNode* conv2d_nodes);
};

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_OPTIMIZER_OPTIONS_RESOLVE_TRANSFORMER_H_
