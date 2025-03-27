#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_

#include <string>

namespace blink::webnn_optimizer {

#define WEBNN_OPTIMIZER_SPEC_OPERATION_LIST(V) \
  V(ArgMinMax)                                 \
  V(BatchNormalization)                        \
  V(Clamp)                                     \
  V(Concat)                                    \
  V(Conv2d)                                    \
  V(CumulativeSum)                             \
  V(DequantizeLinear)                          \
  V(ElementWiseBinary)                         \
  V(ElementWiseUnary)                          \
  V(Elu)                                       \
  V(Expand)                                    \
  V(Gather)                                    \
  V(GatherElements)                            \
  V(GatherND)                                  \
  V(Gelu)                                      \
  V(Gemm)                                      \
  V(Gru)                                       \
  V(GruCell)                                   \
  V(HardSigmoid)                               \
  V(HardSwish)                                 \
  V(InstanceNormalization)                     \
  V(LayerNormalization)                        \
  V(LeakyRelu)                                 \
  V(Linear)                                    \
  V(Lstm)                                      \
  V(LstmCell)                                  \
  V(Matmul)                                    \
  V(Pad)                                       \
  V(Pool2d)                                    \
  V(Prelu)                                     \
  V(QuantizeLinear)                            \
  V(Reduce)                                    \
  V(Relu)                                      \
  V(Resample2d)                                \
  V(Reshape)                                   \
  V(Reverse)                                   \
  V(ScatterElements)                           \
  V(ScatterND)                                 \
  V(Sigmoid)                                   \
  V(Slice)                                     \
  V(Softmax)                                   \
  V(Softplus)                                  \
  V(Softsign)                                  \
  V(Split)                                     \
  V(Tanh)                                      \
  V(Tile)                                      \
  V(Transpose)                                 \
  V(Triangular)                                \
  V(Where)

enum class OpKind {
  // None
  kNone,
// spec Op
#define ENUM(op) k##op,
  WEBNN_OPTIMIZER_SPEC_OPERATION_LIST(ENUM)
#undef ENUM
  // input and constant
  kConstant,
  kInput,
};

inline std::string OpKind2String(OpKind kind) {
  switch (kind) {
    case OpKind::kNone:
      return "None";
    case OpKind::kConstant:
      return "Constant";
    case OpKind::kInput:
      return "Input";

#define CASE(op)      \
  case OpKind::k##op: \
    return #op;
      WEBNN_OPTIMIZER_SPEC_OPERATION_LIST(CASE)
#undef CASE
  }
}

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_
