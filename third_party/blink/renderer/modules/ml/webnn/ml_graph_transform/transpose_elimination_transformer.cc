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

MLOperand* TransposeEliminationTransformer::HandleTranspose(
    MLOperator* transpose) {
  auto* sub_graph_output_operand = transpose->Outputs()[0].Get();
  auto* input_operand = transpose->Inputs()[0].Get();

  auto dep_op_size = input_operand->DependentOperators().size();
  // todo, this is not necessary
  if (dep_op_size != 1) {
    return sub_graph_output_operand;
  }

  DCHECK(input_operand->DependentOperators().Contains(transpose));

  if (input_operand->Kind() != webnn::mojom::blink::Operand::Kind::kOutput) {
    return sub_graph_output_operand;
  }

  auto* front_transpose = const_cast<MLOperator*>(input_operand->Operator());

  if (front_transpose->Kind() !=
      webnn::mojom::internal::Operation_Data::Operation_Tag::kTranspose) {
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

  for (auto& [op, index] : sub_graph_output_operators_to_update) {
    Connect(sub_graph_input_operand, op, index);
  }

  removed_operators_.insert(transpose);
  removed_operators_.insert(front_transpose);
  printf("Successfully eliminate transpose operator\n");
  return sub_graph_input_operand;
}

}  // namespace blink
