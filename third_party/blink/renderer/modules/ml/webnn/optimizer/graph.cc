#include "third_party/blink/renderer/modules/ml/webnn/optimizer/graph.h"

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/ml_operator_converter.h"

namespace blink_mojom = webnn::mojom::blink;
namespace blink::webnn_optimizer {

// static
Graph* Graph::BuildGraphFromML(
    const MLNamedOperands& named_outputs,
    const webnn::ContextProperties& context_properties) {
  HeapHashMap<Member<const MLOperand>, std::pair<Node*, wtf_size_t>>
      output_operand_to_node_and_index;
  HeapHashMap<Member<const MLOperand>, Node*> input_operand_to_node;
  HeapHashMap<Member<const MLOperand>, Node*> const_operand_to_node;

  HeapVector<Member<const MLOperator>>* topologically_sorted_operators =
      GetOperatorsInTopologicalOrder(named_outputs);

  Graph* graph = MakeGarbageCollected<Graph>(&context_properties);

  for (const auto& current_operator : *topologically_sorted_operators) {
    Node* current_node = ConvertMLOperatorToNode(current_operator);

    for (wtf_size_t i = 0; i < current_operator->Outputs().size(); i++) {
      output_operand_to_node_and_index.insert(
          current_operator->Outputs()[i].Get(), std::pair{current_node, i});
    }

    for (wtf_size_t i = 0; i < current_operator->Inputs().size(); ++i) {
      const auto input_operand = current_operator->Inputs()[i];

      switch (input_operand->Kind()) {
        case webnn::mojom::Operand_Kind::kInput: {
          Node* input_node = nullptr;

          if (input_operand_to_node.Contains(input_operand)) {
            input_node = input_operand_to_node.at(input_operand);
          } else {
            input_node = MakeGarbageCollected<InputNode>();
            input_node->SetOperandDescriptors(
                {{input_operand->Name(), input_operand->Descriptor()}});
            input_operand_to_node.insert(input_operand, input_node);

            graph->inputs_.push_back(input_node);
          }
          Edge::Connect(input_node, 0, current_node, i);
          break;
        }
        case webnn::mojom::Operand_Kind::kConstant: {
          Node* const_node = nullptr;

          if (const_operand_to_node.Contains(input_operand)) {
            const_node = const_operand_to_node.at(input_operand);
          } else {
            const_node = MakeGarbageCollected<ConstantNode>();
            const_node->SetOperandDescriptors(
                {{input_operand->Name(), input_operand->Descriptor()}});
            const_operand_to_node.insert(input_operand, const_node);
          }
          Edge::Connect(const_node, 0, current_node, i);
          break;
        }
        case webnn::mojom::Operand_Kind::kOutput: {
          auto [node, idx] = output_operand_to_node_and_index.at(input_operand);
          // The node must exist because of the topological order.
          CHECK(node);
          Edge::Connect(node, idx, current_node, i);
          break;
        }

        default:
          NOTREACHED() << "Invalid Operand Kind\n";
      }
    }
  }

  for (auto [name, output] : named_outputs) {
    auto [node, idx] = output_operand_to_node_and_index.at(output.Get());
    graph->outputs_.push_back(node);
  }

  return graph;
}

void Graph::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
}

void Graph::Print() const {
  auto sorted = TopologicalSort();
  int id = 0;
  for (auto node : sorted) {
    node->SetId(++id);
    node->Print();
  }
}

HeapVector<Node*> Graph::TopologicalSort() const {
  HeapVector<Node*> stack;
  for (auto output : outputs_) {
    stack.push_back(output);
  }
  HeapHashSet<Member<Node>> visited;
  HeapVector<Node*> sorted;

  while (!stack.empty()) {
    Node* node = stack.back();
    if (visited.Contains(node)) {
      stack.pop_back();
      continue;
    }

    bool all_inputs_visited = true;
    for (auto input : node->GetInputNodes()) {
      if (input && !visited.Contains(input)) {
        stack.push_back(input);
        all_inputs_visited = false;
      }
    }

    if (all_inputs_visited) {
      sorted.push_back(node);
      visited.insert(node);
      stack.pop_back();
    }
  }
  return sorted;
}

struct NodeOutputPort {
  Node* node;
  wtf_size_t index;
};

template <bool node_is_graph_output = false>
inline void MojomOperandCreationHelper(
    blink_mojom::GraphInfoPtr& graph_info,
    std::map<const NodeOutputPort, uint64_t>& output_port_to_id_map,
    uint64_t& id,
    Node* node) {
  webnn::mojom::Operand_Kind kind;

  if constexpr (node_is_graph_output) {
    kind = webnn::mojom::Operand_Kind::kOutput;
  } else {
    if (node->op_kind() == OpKind::kInput) {
      kind = webnn::mojom::Operand_Kind::kInput;
    } else if (node->op_kind() == OpKind::kConstant) {
      kind = webnn::mojom::Operand_Kind::kConstant;
    } else {
      kind = webnn::mojom::Operand_Kind::kOutput;
    }
  }

  for (wtf_size_t index = 0; index < node->GetOutputPorts().size(); ++index) {
    auto [name, desc] = node->GetOperandNameAndDescriptors()[index];
    auto operand = blink_mojom::Operand::New();
    operand->kind = kind;
    operand->descriptor = desc;
    operand->name = name;
    output_port_to_id_map[{node, index}] = ++id;
    graph_info->id_to_operand_map.insert(id, std::move(operand));
    if constexpr (node_is_graph_output) {
      graph_info->output_operands.push_back(id);
    } else {
      if (kind == webnn::mojom::Operand_Kind::kInput) {
        graph_info->input_operands.push_back(id);
      } else if (kind == webnn::mojom::Operand_Kind::kConstant) {
        auto constant_node = static_cast<ConstantNode*>(node);
        // graph_info->constant_operand_ids_to_handles.insert(id, constant_node.handle()) );
      }
    }
  }
}

// inline void MojomOperationCreationHelper(
//     blink_mojom::GraphInfoPtr& graph_info,
//     std::map<const blink_mojom::Operand*, uint64_t>& output_port_to_id_map,
//     uint64_t& id,
//     const Node* node) {
//   switch (node->op_kind()) {
//     case OpKind::kInput:
//     case OpKind::kConstant:
//       NOTREACHED() << "Input and Constant is not operator in mojom";

//     case OpKind::kArgMinMax: {
//       auto mojom_op = blink_mojom::ArgMinMax::New();
//       auto argminmax_node = static_cast<const ArgMinMaxNode*>(node);
//       mojom_op->kind = argminmax_node->kind;
//       mojom_op->axis = argminmax_node->axis;
//       mojom_op->keep_dimensions = argminmax_node->keep_dimensions;
//       mojom_op->label = argminmax_node->GetLabel();

//       // mojom_op->input_operand_id =
//       // operand_to_id_map.at(argminmax_node->input());
//       break;
//     }

//     default:
//       NOTREACHED() << "Invalid OpKind";
//   }
// }

webnn::mojom::blink::GraphInfoPtr Graph::ToMojom() const {
  auto graph_info = blink_mojom::GraphInfo::New();
  uint64_t id = 0;

  std::map<const NodeOutputPort, uint64_t> output_port_to_id_map;
  for (auto node : outputs_) {
    MojomOperandCreationHelper<true>(graph_info, output_port_to_id_map, id,
                                     node);
  }

  auto sorted = TopologicalSort();
  for (wtf_size_t i = 0; sorted.size(); ++i) {
    bool is_input_or_constant = true;
    auto node = sorted[i];

    MojomOperandCreationHelper(graph_info, output_port_to_id_map, id, node);

    switch (node->op_kind()) {
      case OpKind::kInput:
      case OpKind::kConstant: {
        MojomOperandCreationHelper(graph_info, output_port_to_id_map, id, node);
        break;
      }
      default:
        is_input_or_constant = false;
        break;
    }
    if (is_input_or_constant) {
      // Input and Constant is not operator in mojom
      continue;
    }
  }

  return graph_info;
}

}  // namespace blink::webnn_optimizer
