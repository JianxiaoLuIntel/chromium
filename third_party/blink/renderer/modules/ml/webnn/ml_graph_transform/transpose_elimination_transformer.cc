#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/transpose_elimination_transformer.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

void TransposeEliminationTransformer::Trace(Visitor* visitor) const {
  MLGraphTransformer::Trace(visitor);
  visitor->Trace(removed_operators_);
}

void TransposeEliminationTransformer::Transform(
    MLNamedOperands& named_outputs) {
  DCHECK(removed_operators_.empty());

  auto* sorted_operators = GetOperatorsInTopologicalOrder(named_outputs);

  HeapHashSet<Member<const MLOperator>> graph_output_operators;
  for (auto& named_output : named_outputs) {
    auto* output_operand = named_output.second.Get();
    graph_output_operators.insert(output_operand->Operator());
  }

  for (auto& op : *sorted_operators) {
    if (removed_operators_.Contains(op)) {
      continue;
    }

    MLOperand* original_operand = op->Outputs()[0].Get();
    MLOperand* updated_operand = original_operand;

    if (op->Kind() ==
        webnn::mojom::internal::Operation_Data::Operation_Tag::kTranspose) {
      updated_operand = HandleTranspose(const_cast<MLOperator*>(op.Get()));
    }

    // The handled operator is a graph output, update named_outputs.
    if (updated_operand != original_operand &&
        graph_output_operators.Contains(op)) {
      for (auto& named_output : named_outputs) {
        if (named_output.second.Get() == original_operand) {
          named_output.second = updated_operand;
        }
      }
    }
  }
  removed_operators_.clear();
}

bool IsLayoutAgnosticNode(MLOperator* node) {
  switch (node->Kind()) {
    case webnn::mojom::internal::Operation_Data::Operation_Tag::kClamp:
      return true;
    default:
      return false;
  }
}

// Skip la_nodes(layout agnostic nodes) and find the front transpose
// For example
// node0 -> transpose0 -> clamp0 -> clamp1 -> transpose1 -> node1
// Can be eliminated to:
// node0 -> clamp0 -> clamp1 -> node1
void TryFindEliminatableFrontTranspose(MLOperator* transpose,
                                       MLOperator*& front_transpose,
                                       MLOperator*& la_node_front,
                                       MLOperator*& la_node_back) {
  DCHECK_EQ(front_transpose, nullptr);
  DCHECK_EQ(la_node_front, nullptr);
  DCHECK_EQ(la_node_back, nullptr);

  if (transpose->Inputs()[0]->Kind() != webnn::mojom::Operand_Kind::kOutput) {
    return;
  }

  MLOperator* cur_node =
      const_cast<MLOperator*>(transpose->Inputs()[0].Get()->Operator());

  while (true) {
    if (cur_node->Outputs().size() != 1 || cur_node->Inputs().size() != 1 ||
        cur_node->Outputs()[0]->DependentOperators().size() != 1) {
      break;
    }

    if (cur_node->Kind() ==
        webnn::mojom::internal::Operation_Data::Operation_Tag::kTranspose) {
      front_transpose = (cur_node);
      break;
    }

    if (IsLayoutAgnosticNode(cur_node)) {
      if (la_node_back == nullptr) {
        DCHECK_EQ(la_node_front, nullptr);
        la_node_back = cur_node;
        la_node_front = cur_node;
      } else {
        la_node_front = cur_node;
      }

      if (cur_node->Inputs()[0]->Kind() !=
          webnn::mojom::Operand_Kind::kOutput) {
        break;
      }

      cur_node =
          const_cast<MLOperator*>(cur_node->Inputs()[0].Get()->Operator());
      continue;
    }
    break;
  }
}

MLOperand* TransposeEliminationTransformer::HandleTranspose(
    MLOperator* transpose) {
  auto* sub_graph_output_operand = transpose->Outputs()[0].Get();
  auto* input_operand = transpose->Inputs()[0].Get();

  MLOperator* front_transpose = nullptr;
  MLOperator* la_node_front = nullptr;
  MLOperator* la_node_back = nullptr;

  TryFindEliminatableFrontTranspose(transpose, front_transpose, la_node_front,
                                    la_node_back);

  if (front_transpose == nullptr) {
    return sub_graph_output_operand;
  }

  auto* options = static_cast<const MLTransposeOptions*>(transpose->Options());
  auto* front_options =
      static_cast<const MLTransposeOptions*>(front_transpose->Options());

  wtf_size_t rank = input_operand->Rank();
  auto permutation =
      front_options->getPermutationOr(CreateDefaultPermutation(rank));
  auto front_permutation =
      options->getPermutationOr(CreateDefaultPermutation(rank));

  if (permutation.size() != front_permutation.size()) {
    return sub_graph_output_operand;
  }

  auto* sub_graph_input_operand = front_transpose->Inputs()[0].Get();

  Disconnect(sub_graph_input_operand, front_transpose, 0);

  HeapVector<std::pair<MLOperator*, int>> sub_graph_output_operators_to_update;

  auto sub_graph_output_operators =
      sub_graph_output_operand->DependentOperators();
  for (auto& sub_graph_output_operator : sub_graph_output_operators) {
    MLOperator* output_op =
        const_cast<MLOperator*>(sub_graph_output_operator.Get());
    int disconnect_index = Disconnect(sub_graph_output_operand, output_op);
    sub_graph_output_operators_to_update.push_back(
        std::pair{output_op, disconnect_index});
  }

  if (la_node_back == nullptr) {
    DCHECK_EQ(la_node_front, nullptr);
    for (auto& [op, index] : sub_graph_output_operators_to_update) {
      Connect(sub_graph_input_operand, op, index);
    }
  } else {
    DCHECK_NE(la_node_front, nullptr);
    Disconnect(front_transpose, 0, la_node_front, 0);
    Disconnect(la_node_back, 0, transpose, 0);

    Connect(sub_graph_input_operand, la_node_front, 0);

    for (auto& [op, index] : sub_graph_output_operators_to_update) {
      Connect(la_node_back, 0, op, index);
    }

    // update la_nodes operand descriptors (shape)
    auto std_shape = la_node_front->Inputs()[0]->Shape();
    Vector<uint32_t> shape(std_shape.size());
    for (size_t i = 0; i < std_shape.size(); ++i) {
      shape[i] = std_shape[i];
    }

    for (MLOperator* cur_node = la_node_back;;
         cur_node =
             const_cast<MLOperator*>(cur_node->Inputs()[0].Get()->Operator())) {
      auto* new_operand = CloneResetShape(cur_node->Outputs()[0], shape);
      ReplaceOperand(cur_node->Outputs()[0], new_operand);

      if (cur_node == la_node_front) {
        break;
      }
    }
  }

  removed_operators_.insert(transpose);
  removed_operators_.insert(front_transpose);
  printf("Successfully eliminate transpose operator\n");
  return sub_graph_input_operand;
}

}  // namespace blink
