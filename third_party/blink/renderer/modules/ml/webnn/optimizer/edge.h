#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_EDGE_H
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_EDGE_H

#include "third_party/blink/renderer/platform/heap/garbage_collected.h"

namespace blink::webnn_optimizer {

class Node;

class Edge : public GarbageCollected<Edge> {
 public:
  static void Connect(Node* from_node,
                      size_t from_index,
                      Node* to_node,
                      size_t to_index);

  Edge(Node* from_node, size_t from_index, Node* to_node, size_t to_index);

  void Trace(Visitor* visitor) const;

  Node* FromNode() const { return from_node_; }
  size_t FromIndex() const { return from_index_; }

  Node* ToNode() const { return to_node_; }
  size_t ToIndex() const { return to_index_; }

 private:
  bool operator==(const Edge& other) const;
  bool operator!=(const Edge& other) const;

  cppgc::Member<Node> from_node_;
  size_t from_index_;
  cppgc::Member<Node> to_node_;
  size_t to_index_;
};

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_EDGE_H