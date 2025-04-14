#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_TRANSPOSE_ELIMINATION_TRANSFORMER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_TRANSPOSE_ELIMINATION_TRANSFORMER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/ml_graph_transformer.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_set.h"

namespace blink {

class TransposeEliminationTransformer : public MLGraphTransformer {
 public:
  explicit TransposeEliminationTransformer(MLGraphBuilder* graph_builder)
      : MLGraphTransformer(graph_builder) {}

  void Trace(Visitor* visitor) const;

  void Transform(MLNamedOperands& named_outputs) override;

 private:
  MLOperand* HandleTranspose(MLOperator* transpose);

  HeapHashSet<Member<MLOperator>> removed_operators_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TRANSFORM_TRANSPOSE_ELIMINATION_TRANSFORMER_H_
