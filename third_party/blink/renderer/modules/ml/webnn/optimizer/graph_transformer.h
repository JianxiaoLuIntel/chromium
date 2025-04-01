#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/graph.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"

namespace blink::webnn_optimizer {
class GraphTransformer : public GarbageCollected<GraphTransformer> {
 public:
  GraphTransformer() = default;
  virtual ~GraphTransformer() = default;
  void Transform(Graph* graph) {
    graph_ = graph;
    TransformInternal();
    graph_ = nullptr;
  }

  virtual void Trace(Visitor* visitor) const { visitor->Trace(graph_); }

 protected:
  Member<Graph> graph_;
  virtual void TransformInternal() = 0;
};
}  // namespace blink::webnn_optimizer
#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
