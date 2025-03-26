#include "third_party/blink/renderer/modules/ml/webnn/optimizer/graph.h"

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/optimizer/ml_operator_converter.h"

namespace blink::webnn_optimizer {

// static
Graph* Graph::BuildGraphFromML(
    const MLNamedOperands& named_outputs,
    const webnn::ContextProperties& context_properties) {
  HeapVector<Node*> outputs;
  HeapHashMap<Member<const MLOperand>, std::pair<Node*, wtf_size_t>>
      output_operand_to_node_and_index;
  HeapHashMap<Member<const MLOperand>, Node*> input_operand_to_node;
  HeapHashMap<Member<const MLOperand>, Node*> const_operand_to_node;

  HeapVector<Member<const MLOperator>>* topologically_sorted_operators =
      GetOperatorsInTopologicalOrder(named_outputs);

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
            input_node->SetOperands({input_operand});
            input_operand_to_node.insert(input_operand, input_node);
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
            const_node->SetOperands({input_operand});
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

  return nullptr;
}

void Graph::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
}
}  // namespace blink::webnn_optimizer