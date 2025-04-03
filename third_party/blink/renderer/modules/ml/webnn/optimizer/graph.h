#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
namespace blink::webnn_optimizer {

class Graph : public GarbageCollected<Graph> {
 public:
  static Graph* BuildGraphFromML(
      const MLNamedOperands& named_outputs,
      const webnn::ContextProperties& context_properties);

  explicit Graph(const webnn::ContextProperties* context_properties)
      : context_properties_(context_properties) {}

  void Trace(Visitor* visitor) const;

  void Print() const;

  HeapVector<Node*> TopologicalSort() const;

  base::raw_ptr<const webnn::ContextProperties> GetContextProperties() const {
    return context_properties_;
  }

  webnn::mojom::blink::GraphInfoPtr ToMojom() const;

 private:
  HeapVector<Member<Node>> inputs_;
  HeapVector<Member<Node>> outputs_;

  const base::raw_ptr<const webnn::ContextProperties> context_properties_;
};

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_GRAPH_H_
