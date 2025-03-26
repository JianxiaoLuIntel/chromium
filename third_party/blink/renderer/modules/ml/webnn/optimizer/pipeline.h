#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_PIPELINE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_PIPELINE_H_

#include "graph_optimizer.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/base.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
#include "third_party/blink/renderer/platform/heap/member.h"

namespace blink::webnn_optimizer {

class OptimizePipeline : public GarbageCollected<OptimizePipeline> {
  explicit OptimizePipeline(ContextKind kind);

  void Trace(Visitor* visitor) const;

 private:
  void RegisterAllOptimizer();
  HeapVector<Member<GraphOptimizer>> optimizers_;
  ContextKind kind_;
};
}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_PIPELINE_H_