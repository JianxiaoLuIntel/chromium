#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/ml_graph_printer.h"

#include <iostream>

#include "services/webnn/webnn_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

void Print(MLNamedOperands& named_outputs) {
  auto* sorted_operators = GetOperatorsInTopologicalOrder(named_outputs);

  size_t id = 0;

  HeapHashMap<Member<const MLOperand>, size_t> input_constant_operand_ids;

  HeapHashMap<Member<const MLOperator>, size_t> operator_ids;

  for (auto& op : *sorted_operators) {
    for (auto& input : op->Inputs()) {
      if (input->Kind() == webnn::mojom::blink::Operand::Kind::kInput) {
        if (!input_constant_operand_ids.Contains(input)) {
          input_constant_operand_ids.insert(input, ++id);
          std::cout << "#" << id << " Input: " << input->Name() << "\n";
        }
      } else if (input->Kind() == webnn::mojom::Operand_Kind::kConstant) {
        if (!input_constant_operand_ids.Contains(input)) {
          input_constant_operand_ids.insert(input, ++id);
          std::cout << "#" << id << " Constant" << "\n";
        }
      }
    }

    String opname = MLOperator::OperatorKindToString(op->Kind(), op->SubKind());
    operator_ids.insert(op, ++id);
    std::cout << "#" << id << " " << opname.Utf8() << " (";
    for (auto& input : op->Inputs()) {
      if (input->Kind() == webnn::mojom::Operand_Kind::kInput ||
          input->Kind() == webnn::mojom::Operand_Kind::kConstant) {
        std::cout << "#" << input_constant_operand_ids.at(input) << " ";
      } else {
        std::cout << "#" << operator_ids.at(input->Operator()) << " ";
      }
    }
    std::cout << ")\n";
  }
}

}  // namespace blink
