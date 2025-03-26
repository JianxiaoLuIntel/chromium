#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
namespace blink::webnn_optimizer {

class Graph : GarbageCollected<Graph> {
 public:
  static Graph* BuildGraphFromML(
      const MLNamedOperands& named_outputs,
      const webnn::ContextProperties& context_properties);

  void Trace(Visitor* visitor) const;

 private:
  HeapVector<Member<Node>> inputs_;
  HeapVector<Member<Node>> outputs_;
};

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_