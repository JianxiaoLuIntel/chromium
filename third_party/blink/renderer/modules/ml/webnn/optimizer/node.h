#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPERATION_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPERATION_H_

#include "services/webnn/public/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/edge.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/opset.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/garbage_collected.h"
#include "third_party/blink/renderer/platform/wtf/hash_traits.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink::webnn_optimizer {
class Node : public GarbageCollected<Node> {
 public:
  virtual void Trace(Visitor* visitor) const;

  virtual OpKind op_kind() const { return OpKind::kNone; }

  Node* GetInputNode(int index) const { return inputs_[index]->FromNode(); }

  void SetLabel(const String& label) { label_ = label; }
  void SetOperands(const HeapVector<Member<MLOperand>>& operands) {
    operands_ = operands;
  }

  virtual ~Node() = default;

 protected:
  Node() = default;
  HeapVector<Member<Edge>> inputs_;
  HeapVector<HeapVector<Member<Edge>>> output_ports_;
  HeapVector<Member<MLOperand>> operands_;
  String label_;

  friend class Edge;
};

// For some operations, the number of input/output ports is not fixed.
static constexpr int kDynamicIOCount = -1;

template <typename Derived, int InputCount, int OutputPortCount>
class NodeT : public Node {
 public:
  void SetInputNum(int input_num) {
    static_assert(InputCount == kDynamicIOCount,
                  "Only node with dynamic input count can be set manually.");
    inputs_.resize(input_num);
  }

  void SetOutputPortNum(int output_port_num) {
    static_assert(
        OutputPortCount == kDynamicIOCount,
        "Only node with dynamic output port count can be set manually.");
    output_ports_.resize(output_port_num);
  }

  template <typename U, typename = void>
  struct has_kind : std::false_type {};

  template <typename U>
  struct has_kind<U, std::void_t<decltype(U::kind)>> : std::true_type {};

  static constexpr bool has_kind_v = has_kind<Derived>::value;

 protected:
  NodeT() {
    if (InputCount != kDynamicIOCount) {
      inputs_.resize(InputCount);
    }
    if (OutputPortCount != kDynamicIOCount) {
      output_ports_.resize(OutputPortCount);
    }
  }
};

template <typename T>
inline bool Is(const Node* node) {
  return node->op_kind() == T::StaticOpKind();
}

template <typename T>
inline T* Cast(const Node* node) {
  CHECK(Is<T>(node));
  return static_cast<T*>(node);
}

template <typename T>
inline T* TryCast(const Node* node) {
  if (Is<T>(node)) {
    return static_cast<T*>(node);
  }
  return nullptr;
}

class InputNode : public NodeT<InputNode, 0, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kInput; }
  InputNode() = default;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kInput; }
};

class ConstantNode : public NodeT<ConstantNode, 0, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kConstant; }
  ConstantNode() = default;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kConstant; }
};

// below is auto gen code

/*['ArgMinMax', 'BatchNormalization', 'Clamp', 'Concat', 'Conv2d',
 * 'CumulativeSum', 'DequantizeLinear', 'ElementWiseBinary', 'Elu',
 * 'ElementWiseUnary', 'Expand', 'Gather', 'GatherElements', 'GatherND', 'Gelu',
 * 'Gemm', 'Gru', 'GruCell', 'HardSigmoid', 'HardSwish', 'LayerNormalization',
 * 'InstanceNormalization', 'LeakyRelu', 'Linear', 'Lstm', 'LstmCell', 'Matmul',
 * 'Pad', 'Pool2d', 'Prelu', 'QuantizeLinear', 'Reduce', 'Relu', 'Resample2d',
 * 'Reshape', 'Reverse', 'ScatterElements', 'ScatterND', 'Sigmoid', 'Slice',
 * 'Softmax', 'Softplus', 'Softsign', 'Split', 'Tanh', 'Tile', 'Transpose',
 * 'Triangular', 'Where']*/

class ArgMinMaxNode : public NodeT<ArgMinMaxNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kArgMinMax; }
  ArgMinMaxNode() = default;
  Node* input() const { return GetInputNode(0); }

  webnn::mojom::blink::ArgMinMax::Kind kind;
  uint32_t axis;
  bool keep_dimensions;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kArgMinMax; }
};

class BatchNormalizationNode : public NodeT<BatchNormalizationNode, 5, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kBatchNormalization; }
  BatchNormalizationNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* mean() const { return GetInputNode(1); }
  Node* variance() const { return GetInputNode(2); }
  Node* scale() const { return GetInputNode(3); }
  Node* bias() const { return GetInputNode(4); }

  uint32_t axis;
  float epsilon;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kBatchNormalization; }
};

class ClampNode : public NodeT<ClampNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kClamp; }
  ClampNode() = default;
  Node* input() const { return GetInputNode(0); }

  float min_value;
  float max_value;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kClamp; }
};

class ConcatNode : public NodeT<ConcatNode, kDynamicIOCount, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kConcat; }
  ConcatNode() = default;

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kConcat; }
};

class Conv2dNode : public NodeT<Conv2dNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kConv2d; }
  Conv2dNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* filter() const { return GetInputNode(1); }
  Node* bias() const { return GetInputNode(2); }

  webnn::mojom::blink::Conv2d::Kind kind;
  webnn::mojom::blink::Padding2d padding;
  webnn::mojom::blink::Size2d strides;
  webnn::mojom::blink::Size2d dilations;
  uint32_t groups;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kConv2d; }
};

class CumulativeSumNode : public NodeT<CumulativeSumNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kCumulativeSum; }
  CumulativeSumNode() = default;
  Node* input() const { return GetInputNode(0); }

  uint32_t axis;
  bool exclusive;
  bool reversed;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kCumulativeSum; }
};

class DequantizeLinearNode : public NodeT<DequantizeLinearNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kDequantizeLinear; }
  DequantizeLinearNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* scale() const { return GetInputNode(1); }
  Node* zero_point() const { return GetInputNode(2); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kDequantizeLinear; }
};

class ElementWiseBinaryNode : public NodeT<ElementWiseBinaryNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kElementWiseBinary; }
  ElementWiseBinaryNode() = default;
  Node* lhs() const { return GetInputNode(0); }
  Node* rhs() const { return GetInputNode(1); }

  webnn::mojom::blink::ElementWiseBinary::Kind kind;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kElementWiseBinary; }
};

class ElementWiseUnaryNode : public NodeT<ElementWiseUnaryNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kElementWiseUnary; }
  ElementWiseUnaryNode() = default;
  Node* input() const { return GetInputNode(0); }

  webnn::mojom::blink::ElementWiseUnary::Kind kind;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kElementWiseUnary; }
};

class ExpandNode : public NodeT<ExpandNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kExpand; }
  ExpandNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kExpand; }
};

class InstanceNormalizationNode
    : public NodeT<InstanceNormalizationNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kInstanceNormalization; }
  InstanceNormalizationNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* scale() const { return GetInputNode(1); }
  Node* bias() const { return GetInputNode(2); }

  float epsilon;
  webnn::mojom::blink::InputOperandLayout layout;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kInstanceNormalization; }
};

class MatmulNode : public NodeT<MatmulNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kMatmul; }
  MatmulNode() = default;
  Node* a() const { return GetInputNode(0); }
  Node* b() const { return GetInputNode(1); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kMatmul; }
};

class PadNode : public NodeT<PadNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kPad; }
  PadNode() = default;
  Node* input() const { return GetInputNode(0); }

  Vector<uint32_t> beginning_padding;
  Vector<uint32_t> ending_padding;
  webnn::mojom::blink::PaddingModePtr mode;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kPad; }
};

class ReduceNode : public NodeT<ReduceNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kReduce; }
  ReduceNode() = default;
  Node* input() const { return GetInputNode(0); }

  webnn::mojom::blink::Reduce::Kind kind;
  Vector<uint32_t> axes;
  bool keep_dimensions;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kReduce; }
};

class Pool2dNode : public NodeT<Pool2dNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kPool2d; }
  Pool2dNode() = default;
  Node* input() const { return GetInputNode(0); }

  webnn::mojom::blink::Pool2d::Kind kind;
  webnn::mojom::blink::Size2d window_dimensions;
  webnn::mojom::blink::Padding2d padding;
  webnn::mojom::blink::Size2d strides;
  webnn::mojom::blink::Size2d dilations;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kPool2d; }
};

class SliceNode : public NodeT<SliceNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSlice; }
  SliceNode() = default;
  Node* input() const { return GetInputNode(0); }

  Vector<webnn::mojom::blink::Range> ranges;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSlice; }
};

class EluNode : public NodeT<EluNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kElu; }
  EluNode() = default;
  Node* input() const { return GetInputNode(0); }

  float alpha;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kElu; }
};

class GatherNode : public NodeT<GatherNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGather; }
  GatherNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* indices() const { return GetInputNode(1); }

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGather; }
};

class GatherElementsNode : public NodeT<GatherElementsNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGatherElements; }
  GatherElementsNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* indices() const { return GetInputNode(1); }

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGatherElements; }
};

class GatherNDNode : public NodeT<GatherNDNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGatherND; }
  GatherNDNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* indices() const { return GetInputNode(1); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGatherND; }
};

class GeluNode : public NodeT<GeluNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGelu; }
  GeluNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGelu; }
};

class GruNode : public NodeT<GruNode, 6, kDynamicIOCount> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGru; }
  GruNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* weight() const { return GetInputNode(1); }
  Node* recurrent_weight() const { return GetInputNode(2); }
  Node* bias() const { return GetInputNode(3); }
  Node* recurrent_bias() const { return GetInputNode(4); }
  Node* initial_hidden_state() const { return GetInputNode(5); }

  uint32_t steps;
  uint32_t hidden_size;
  bool reset_after;
  bool return_sequence;
  webnn::mojom::blink::RecurrentNetworkDirection direction;
  webnn::mojom::blink::GruWeightLayout layout;
  Vector<webnn::mojom::blink::RecurrentNetworkActivation> activations;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGru; }
};

class GruCellNode : public NodeT<GruCellNode, 6, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGruCell; }
  GruCellNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* weight() const { return GetInputNode(1); }
  Node* recurrent_weight() const { return GetInputNode(2); }
  Node* hidden_state() const { return GetInputNode(3); }
  Node* bias() const { return GetInputNode(4); }
  Node* recurrent_bias() const { return GetInputNode(5); }

  uint32_t hidden_size;
  bool reset_after;
  webnn::mojom::blink::GruWeightLayout layout;
  Vector<webnn::mojom::blink::RecurrentNetworkActivation> activations;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGruCell; }
};

class GemmNode : public NodeT<GemmNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kGemm; }
  GemmNode() = default;
  Node* a() const { return GetInputNode(0); }
  Node* b() const { return GetInputNode(1); }
  Node* c() const { return GetInputNode(2); }

  float alpha;
  float beta;
  bool a_transpose;
  bool b_transpose;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kGemm; }
};

class HardSigmoidNode : public NodeT<HardSigmoidNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kHardSigmoid; }
  HardSigmoidNode() = default;
  Node* input() const { return GetInputNode(0); }

  float alpha;
  float beta;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kHardSigmoid; }
};

class HardSwishNode : public NodeT<HardSwishNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kHardSwish; }
  HardSwishNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kHardSwish; }
};

class LayerNormalizationNode : public NodeT<LayerNormalizationNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kLayerNormalization; }
  LayerNormalizationNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* scale() const { return GetInputNode(1); }
  Node* bias() const { return GetInputNode(2); }

  Vector<uint32_t> axes;
  float epsilon;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kLayerNormalization; }
};

class LeakyReluNode : public NodeT<LeakyReluNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kLeakyRelu; }
  LeakyReluNode() = default;
  Node* input() const { return GetInputNode(0); }

  float alpha;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kLeakyRelu; }
};

class LinearNode : public NodeT<LinearNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kLinear; }
  LinearNode() = default;
  Node* input() const { return GetInputNode(0); }

  float alpha;
  float beta;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kLinear; }
};

class LstmNode : public NodeT<LstmNode, 8, kDynamicIOCount> {
 public:
  static OpKind StaticOpKind() { return OpKind::kLstm; }
  LstmNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* weight() const { return GetInputNode(1); }
  Node* recurrent_weight() const { return GetInputNode(2); }
  Node* bias() const { return GetInputNode(3); }
  Node* recurrent_bias() const { return GetInputNode(4); }
  Node* peephole_weight() const { return GetInputNode(5); }
  Node* initial_hidden_state() const { return GetInputNode(6); }
  Node* initial_cell_state() const { return GetInputNode(7); }

  uint32_t steps;
  uint32_t hidden_size;
  bool return_sequence;
  webnn::mojom::blink::RecurrentNetworkDirection direction;
  webnn::mojom::blink::LstmWeightLayout layout;
  Vector<webnn::mojom::blink::RecurrentNetworkActivation> activations;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kLstm; }
};

class LstmCellNode : public NodeT<LstmCellNode, 8, kDynamicIOCount> {
 public:
  static OpKind StaticOpKind() { return OpKind::kLstmCell; }
  LstmCellNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* weight() const { return GetInputNode(1); }
  Node* recurrent_weight() const { return GetInputNode(2); }
  Node* hidden_state() const { return GetInputNode(3); }
  Node* cell_state() const { return GetInputNode(4); }
  Node* bias() const { return GetInputNode(5); }
  Node* recurrent_bias() const { return GetInputNode(6); }
  Node* peephole_weight() const { return GetInputNode(7); }

  uint32_t hidden_size;
  webnn::mojom::blink::LstmWeightLayout layout;
  Vector<webnn::mojom::blink::RecurrentNetworkActivation> activations;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kLstmCell; }
};

class PreluNode : public NodeT<PreluNode, 2, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kPrelu; }
  PreluNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* slope() const { return GetInputNode(1); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kPrelu; }
};

class QuantizeLinearNode : public NodeT<QuantizeLinearNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kQuantizeLinear; }
  QuantizeLinearNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* scale() const { return GetInputNode(1); }
  Node* zero_point() const { return GetInputNode(2); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kQuantizeLinear; }
};

class ReluNode : public NodeT<ReluNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kRelu; }
  ReluNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kRelu; }
};

class ReshapeNode : public NodeT<ReshapeNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kReshape; }
  ReshapeNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kReshape; }
};

class ReverseNode : public NodeT<ReverseNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kReverse; }
  ReverseNode() = default;
  Node* input() const { return GetInputNode(0); }

  Vector<uint32_t> axes;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kReverse; }
};

class ScatterElementsNode : public NodeT<ScatterElementsNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kScatterElements; }
  ScatterElementsNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* indices() const { return GetInputNode(1); }
  Node* updates() const { return GetInputNode(2); }

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kScatterElements; }
};

class ScatterNDNode : public NodeT<ScatterNDNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kScatterND; }
  ScatterNDNode() = default;
  Node* input() const { return GetInputNode(0); }
  Node* indices() const { return GetInputNode(1); }
  Node* updates() const { return GetInputNode(2); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kScatterND; }
};

class SigmoidNode : public NodeT<SigmoidNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSigmoid; }
  SigmoidNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSigmoid; }
};

class SoftmaxNode : public NodeT<SoftmaxNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSoftmax; }
  SoftmaxNode() = default;
  Node* input() const { return GetInputNode(0); }

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSoftmax; }
};

class SoftplusNode : public NodeT<SoftplusNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSoftplus; }
  SoftplusNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSoftplus; }
};

class SoftsignNode : public NodeT<SoftsignNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSoftsign; }
  SoftsignNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSoftsign; }
};

class SplitNode : public NodeT<SplitNode, 1, kDynamicIOCount> {
 public:
  static OpKind StaticOpKind() { return OpKind::kSplit; }
  SplitNode() = default;
  Node* input() const { return GetInputNode(0); }

  uint32_t axis;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kSplit; }
};

class TanhNode : public NodeT<TanhNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kTanh; }
  TanhNode() = default;
  Node* input() const { return GetInputNode(0); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kTanh; }
};

class TileNode : public NodeT<TileNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kTile; }
  TileNode() = default;
  Node* input() const { return GetInputNode(0); }

  Vector<uint32_t> repetitions;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kTile; }
};

class TransposeNode : public NodeT<TransposeNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kTranspose; }
  TransposeNode() = default;
  Node* input() const { return GetInputNode(0); }

  Vector<uint32_t> permutation;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kTranspose; }
};

class TriangularNode : public NodeT<TriangularNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kTriangular; }
  TriangularNode() = default;
  Node* input() const { return GetInputNode(0); }

  bool upper;
  int32_t diagonal;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kTriangular; }
};

class Resample2dNode : public NodeT<Resample2dNode, 1, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kResample2d; }
  Resample2dNode() = default;
  Node* input() const { return GetInputNode(0); }

  webnn::mojom::blink::Resample2d::InterpolationMode mode;
  Vector<uint32_t> axes;

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kResample2d; }
};

class WhereNode : public NodeT<WhereNode, 3, 1> {
 public:
  static OpKind StaticOpKind() { return OpKind::kWhere; }
  WhereNode() = default;
  Node* condition() const { return GetInputNode(0); }
  Node* true_value() const { return GetInputNode(1); }
  Node* false_value() const { return GetInputNode(2); }

  void Trace(Visitor* visitor) const override { Node::Trace(visitor); }

  OpKind op_kind() const override { return OpKind::kWhere; }
};

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPERATION_H_