#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_

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

}  // namespace blink::webnn_optimizer

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPTIMIZER_OPSET_H_
