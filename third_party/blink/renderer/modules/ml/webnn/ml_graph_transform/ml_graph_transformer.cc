#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/ml_graph_transformer.h"

#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

// static
void MLGraphTransformer::Disconnect(MLOperator* from,
                                    int from_index,
                                    MLOperator* to,
                                    int to_index) {
  MLOperand* operand = from->outputs_[from_index];
  auto& dependent_operators = operand->dependent_operators_;

  DCHECK(dependent_operators.Contains(to));
  dependent_operators.erase(to);

  DCHECK(to->inputs_[to_index] == operand);
  to->inputs_[to_index] = nullptr;
}

// static
int MLGraphTransformer::Disconnect(MLOperator* from,
                                   int from_index,
                                   MLOperator* to) {
  MLOperand* operand = from->outputs_[from_index];
  auto& dependent_operators = operand->dependent_operators_;

  DCHECK(dependent_operators.Contains(to));
  dependent_operators.erase(to);

  wtf_size_t to_index = to->inputs_.Find(operand);
  DCHECK(to_index != kNotFound);
  to->inputs_[to_index] = nullptr;
  return static_cast<int>(to_index);
}

// static
int MLGraphTransformer::Disconnect(MLOperand* from, MLOperator* to) {
  auto& dependent_operators = from->dependent_operators_;

  DCHECK(dependent_operators.Contains(to));
  dependent_operators.erase(to);

  wtf_size_t to_index = to->inputs_.Find(from);
  DCHECK(to_index != kNotFound);
  to->inputs_[to_index] = nullptr;
  return static_cast<int>(to_index);
}

// static
void MLGraphTransformer::Disconnect(MLOperand* from,
                                    MLOperator* to,
                                    int to_index) {
  auto& dependent_operators = from->dependent_operators_;

  DCHECK(dependent_operators.Contains(to));
  dependent_operators.erase(to);

  DCHECK(to->inputs_[to_index] == from);
  to->inputs_[to_index] = nullptr;
}

// static
void MLGraphTransformer::Connect(MLOperand* from,
                                 MLOperator* to,
                                 int to_index) {
  from->AddDependentOperator(to);

  DCHECK_EQ(to->inputs_[to_index], nullptr);
  to->inputs_[to_index] = from;
}

// static
void MLGraphTransformer::Connect(MLOperator* from,
                                 int from_index,
                                 MLOperator* to,
                                 int to_index) {
  MLOperand* operand = from->outputs_[from_index];
  operand->AddDependentOperator(to);

  DCHECK_EQ(to->inputs_[to_index], nullptr);
  to->inputs_[to_index] = operand;
}

// static
MLOperand* MLGraphTransformer::CloneResetShape(const MLOperand* operand,
                                               const Vector<uint32_t>& shape) {
  auto descriptor = webnn::OperandDescriptor::Create(
      operand->Builder()->GetContext()->GetProperties(), operand->DataType(),
      shape, "");

  MLOperand* clone = MakeGarbageCollected<MLOperand>(
      operand->Builder(), operand->Kind(), descriptor.value());

  clone->operator_ = operand->Operator();
  clone->dependent_operators_ = operand->dependent_operators_;
  return clone;
}

// static
void MLGraphTransformer::ReplaceOperand(MLOperand* old_operand,
                                        MLOperand* new_operand) {
  auto* op = const_cast<MLOperator*>(old_operand->Operator());
  for (auto& output : op->outputs_) {
    if (output == old_operand) {
      output = new_operand;
    }
  }

  auto& deps = old_operand->dependent_operators_;
  for (auto& dep : deps) {
    auto* dep_op = const_cast<MLOperator*>(dep.Get());
    for (auto& input : dep_op->inputs_) {
      if (input == old_operand) {
        input = new_operand;
      }
    }
  }
}

void MLGraphTransformer::Trace(Visitor* visitor) const {
  visitor->Trace(graph_builder_);
}

const ExceptionState MLGraphTransformer::GetExceptionState() {
  auto* isolate = graph_builder_->GetExecutionContext()->GetIsolate();
  return ExceptionState(isolate);
}

}  // namespace blink
