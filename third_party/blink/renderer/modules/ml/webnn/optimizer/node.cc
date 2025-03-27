#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"

#include <iostream>

namespace blink::webnn_optimizer {

void Node::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(output_ports_);
  visitor->Trace(operands_);
}

void Node::Print() const {
  std::string op_kind_str = OpKind2String(op_kind());
  std::cout << "#" << id_ << " " << op_kind_str;
  if (!inputs_.empty()) {
    std::cout << "\t" << "(";
    std::string input_str = "";
    for (auto input_edge : inputs_) {
      if (input_edge == nullptr) {
        // optional input is null
        input_str += "#null,";
      } else {
        auto* input_node = input_edge->FromNode();
        input_str += ("#" + std::to_string(input_node->id_) + ",");
      }
    }
    input_str.pop_back();
    std::cout << input_str << ")";
  }
  std::cout << std::endl;
}

}  // namespace blink::webnn_optimizer
