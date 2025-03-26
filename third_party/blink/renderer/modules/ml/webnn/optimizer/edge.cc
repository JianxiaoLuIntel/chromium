#include "third_party/blink/renderer/modules/ml/webnn/optimizer/edge.h"

#include "cppgc/visitor.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/node.h"

namespace blink::webnn_optimizer {

// static
void Edge::Connect(Node* from_node,
                   size_t from_index,
                   Node* to_node,
                   size_t to_index) {
  auto* edge =
      MakeGarbageCollected<Edge>(from_node, from_index, to_node, to_index);

  DCHECK_GT(from_node->output_ports_.size(), from_index);

#ifdef _DEBUG
  // CHECK the edge is not connected before.
  for (auto& output : from_node->output_ports_[from_index]) {
    DCHECK(*output != *edge);
  }

#endif

  from_node->output_ports_[from_index].push_back(edge);

  DCHECK_GT(to_node->inputs_.size(), to_index);
  DCHECK(to_node->inputs_[to_index] == nullptr);
  to_node->inputs_[to_index] = edge;
}

void Edge::Trace(Visitor* visitor) const {
  visitor->Trace(from_node_);
  visitor->Trace(to_node_);
}

Edge::Edge(Node* from_node, size_t from_index, Node* to_node, size_t to_index)
    : from_node_(from_node),
      from_index_(from_index),
      to_node_(to_node),
      to_index_(to_index) {}

bool Edge::operator==(const Edge& other) const {
  return from_node_ == other.from_node_ && from_index_ == other.from_index_ &&
         to_node_ == other.to_node_ && to_index_ == other.to_index_;
}
bool Edge::operator!=(const Edge& other) const {
  return !(*this == other);
}

}  // namespace blink::webnn_optimizer