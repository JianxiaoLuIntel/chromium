#include "third_party/blink/renderer/modules/ml/webnn/optimizer/pipeline.h"

namespace blink::webnn_optimizer {
OptimizePipeline::OptimizePipeline(ContextKind kind) : kind_(kind) {}

void OptimizePipeline::Trace(Visitor* visitor) const {
  visitor->Trace(optimizers_);
}

void OptimizePipeline::RegisterAllOptimizer() {
  // Register all optimizers based on the context kind.
  switch (kind_) {
    case ContextKind::kDml:
      // Register DML optimizers.
      break;
    case ContextKind::kTflite:
      // Register Tflite optimizers.
      break;
    case ContextKind::kCoreml:
      // Register Coreml optimizers.
      break;
    default:
      NOTREACHED() << "Invalid context kind";
  }
}
}  // namespace blink::webnn_optimizer