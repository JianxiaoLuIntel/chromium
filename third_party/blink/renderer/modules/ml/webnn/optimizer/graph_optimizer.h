#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
namespace blink::webnn_optimizer {
class GraphOptimizer : public GarbageCollected<GraphOptimizer> {
 public:
  GraphOptimizer() = default;
  virtual ~GraphOptimizer() = default;
  virtual void Optimize() = 0;
  virtual void Trace(Visitor* visitor) const;
};
}  // namespace blink::webnn_optimizer
#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_OPTIMIZER_H_
