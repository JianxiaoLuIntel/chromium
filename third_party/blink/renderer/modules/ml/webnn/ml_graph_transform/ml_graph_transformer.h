#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_TRANSFORMER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_TRANSFORMER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

namespace blink {

class MLGraphTransformer : public GarbageCollected<MLGraphTransformer> {
 public:
  explicit MLGraphTransformer(MLGraphBuilder* graph_builder)
      : graph_builder_(graph_builder) {}

  virtual ~MLGraphTransformer() = default;

  static void Disconnect(MLOperator* from,
                         int from_index,
                         MLOperator* to,
                         int to_index);

  static int Disconnect(MLOperator* from, int from_index, MLOperator* to);

  static void Disconnect(MLOperand* from, MLOperator* to, int to_index);

  static void Connect(MLOperand* from, MLOperator* to, int to_index);

  static void Connect(MLOperator* from,
                      int from_index,
                      MLOperator* to,
                      int to_index);

  static MLOperand* CloneResetShape(const MLOperand* operand,
                                    const Vector<uint32_t>& shape);

  static void ReplaceOperand(MLOperand* old_operand, MLOperand* new_operand);

  void Trace(Visitor* visitor) const;

  // Apply the transformation to the given graph.
  virtual void Transform(const MLNamedOperands& named_outputs) = 0;

  const ExceptionState GetExceptionState();

 protected:
  Member<MLGraphBuilder> graph_builder_;
};

}  // namespace blink
#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_ML_GRAPH_TRANSFORMER_H_
