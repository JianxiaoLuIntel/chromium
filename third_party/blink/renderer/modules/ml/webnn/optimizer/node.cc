#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"

namespace blink::webnn_optimizer {

void Node::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(output_ports_);
  visitor->Trace(operands_);
}
}  // namespace blink::webnn_optimizer