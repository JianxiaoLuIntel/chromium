#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_transform/option_expansion_transformer.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_transpose_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink_mojom = webnn::mojom::blink;

namespace blink {

namespace {

webnn::InputOperandLayout BlinkInputOperandLayoutToNative(
    blink::V8MLInputOperandLayout::Enum type) {
  switch (type) {
    case blink::V8MLInputOperandLayout::Enum::kNchw:
      return webnn::InputOperandLayout::kNchw;
    case blink::V8MLInputOperandLayout::Enum::kNhwc:
      return webnn::InputOperandLayout::kNhwc;
  }
}

constexpr std::array<uint32_t, 4> kNchwToNhwcPermutation = {0u, 2u, 3u, 1u};
constexpr std::array<uint32_t, 4> kNhwcToNchwPermutation = {0u, 3u, 1u, 2u};

std::optional<base::span<const uint32_t>> GetInputOperandPermutation(
    blink::V8MLInputOperandLayout::Enum input_layout,
    const webnn::ContextProperties& context_properties) {
  if (BlinkInputOperandLayoutToNative(input_layout) ==
      context_properties.input_operand_layout) {
    return std::nullopt;
  }

  switch (input_layout) {
    case blink::V8MLInputOperandLayout::Enum::kNchw:
      CHECK_EQ(context_properties.input_operand_layout,
               webnn::InputOperandLayout::kNhwc);
      return kNchwToNhwcPermutation;
    case blink::V8MLInputOperandLayout::Enum::kNhwc:
      CHECK_EQ(context_properties.input_operand_layout,
               webnn::InputOperandLayout::kNchw);
      return kNhwcToNchwPermutation;
  }
}

std::optional<base::span<const uint32_t>> GetOutputOperandPermutation(
    blink::V8MLInputOperandLayout::Enum input_layout,
    const webnn::ContextProperties& context_properties) {
  if (BlinkInputOperandLayoutToNative(input_layout) ==
      context_properties.input_operand_layout) {
    return std::nullopt;
  }

  // The output layout is the same as the input layout and so the output
  // needs to have the inverse of the permutation returned by
  // `GetInputOperandPermutation()` applied.
  switch (input_layout) {
    case blink::V8MLInputOperandLayout::Enum::kNchw:
      CHECK_EQ(context_properties.input_operand_layout,
               webnn::InputOperandLayout::kNhwc);
      return kNhwcToNchwPermutation;
    case blink::V8MLInputOperandLayout::Enum::kNhwc:
      CHECK_EQ(context_properties.input_operand_layout,
               webnn::InputOperandLayout::kNchw);
      return kNchwToNhwcPermutation;
  }
}

std::optional<std::array<uint32_t, 4>> GetConv2DFilterPermutation(
    webnn::InputOperandLayout input_layout,
    bool depthwise,
    blink::V8MLConv2dFilterOperandLayout filter_layout) {
  switch (input_layout) {
    case webnn::InputOperandLayout::kNchw:
      // Mojo expects the OIHW layout.
      switch (filter_layout.AsEnum()) {
        case blink::V8MLConv2dFilterOperandLayout::Enum::kOihw:
          return std::nullopt;
        case blink::V8MLConv2dFilterOperandLayout::Enum::kHwio:
          return std::to_array<uint32_t>({3u, 2u, 0u, 1u});
        case blink::V8MLConv2dFilterOperandLayout::Enum::kOhwi:
          return std::to_array<uint32_t>({0u, 3u, 1u, 2u});
        case blink::V8MLConv2dFilterOperandLayout::Enum::kIhwo:
          return std::to_array<uint32_t>({3u, 0u, 1u, 2u});
      }
      break;
    case webnn::InputOperandLayout::kNhwc:
      if (depthwise) {
        // Mojo expects the IHWO layout.
        switch (filter_layout.AsEnum()) {
          case blink::V8MLConv2dFilterOperandLayout::Enum::kOihw:
            return std::to_array<uint32_t>({1u, 2u, 3u, 0u});
          case blink::V8MLConv2dFilterOperandLayout::Enum::kHwio:
            return std::to_array<uint32_t>({2u, 0u, 1u, 3u});
          case blink::V8MLConv2dFilterOperandLayout::Enum::kOhwi:
            return std::to_array<uint32_t>({3u, 1u, 2u, 0u});
          case blink::V8MLConv2dFilterOperandLayout::Enum::kIhwo:
            return std::nullopt;
        }
      } else {
        switch (filter_layout.AsEnum()) {
          // Mojo expects the OHWI layout.
          case blink::V8MLConv2dFilterOperandLayout::Enum::kOihw:
            return std::to_array<uint32_t>({0u, 2u, 3u, 1u});
          case blink::V8MLConv2dFilterOperandLayout::Enum::kHwio:
            return std::to_array<uint32_t>({3u, 0u, 1u, 2u});
          case blink::V8MLConv2dFilterOperandLayout::Enum::kOhwi:
            return std::nullopt;
          case blink::V8MLConv2dFilterOperandLayout::Enum::kIhwo:
            return std::to_array<uint32_t>({3u, 1u, 2u, 0u});
        }
      }
      break;
  }
}

std::optional<std::array<uint32_t, 4>> GetConvTranspose2DFilterPermutation(
    webnn::InputOperandLayout input_layout,
    blink::V8MLConvTranspose2dFilterOperandLayout filter_layout) {
  switch (input_layout) {
    case webnn::InputOperandLayout::kNchw:
      // Mojo expects IOHW layout.
      switch (filter_layout.AsEnum()) {
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kIohw:
          return std::nullopt;
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kHwoi:
          return std::to_array<uint32_t>({3, 2, 0, 1});
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kOhwi:
          return std::to_array<uint32_t>({3u, 0u, 1u, 2u});
      }
      break;
    case webnn::InputOperandLayout::kNhwc:
      // Mojo expects OHWI layout.
      switch (filter_layout.AsEnum()) {
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kIohw:
          return std::to_array<uint32_t>({1u, 2u, 3u, 0u});
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kHwoi:
          return std::to_array<uint32_t>({2u, 0u, 1u, 3u});
        case blink::V8MLConvTranspose2dFilterOperandLayout::Enum::kOhwi:
          return std::nullopt;
      }
      break;
  }
}

bool IsDepthwiseConv2d(const MLOperator* conv2d) {
  const auto* options = static_cast<const MLConv2dOptions*>(conv2d->Options());
  CHECK(options);

  const MLOperand* input = conv2d->Inputs()[0];
  CHECK(input);
  const std::vector<uint32_t>& input_shape = input->Shape();
  CHECK_EQ(input_shape.size(), 4u);
  const MLOperand* output = conv2d->Outputs()[0].Get();
  CHECK(output);
  const std::vector<uint32_t>& output_shape = output->Shape();
  CHECK_EQ(output_shape.size(), 4u);

  uint32_t input_channels, output_channels;
  switch (options->inputLayout().AsEnum()) {
    case blink::V8MLInputOperandLayout::Enum::kNchw:
      input_channels = input_shape[1];
      output_channels = output_shape[1];
      break;
    case blink::V8MLInputOperandLayout::Enum::kNhwc:
      input_channels = input_shape[3];
      output_channels = output_shape[3];
      break;
  }

  const uint32_t groups = base::checked_cast<uint32_t>(options->groups());
  return webnn::IsDepthwiseConv2d(input_channels, output_channels, groups);
}

Vector<uint32_t> PermuteShape(base::span<const uint32_t> shape,
                              base::span<const uint32_t> permutation) {
  wtf_size_t shape_size = base::checked_cast<wtf_size_t>(shape.size());
  Vector<uint32_t> permuted_array(shape_size);

  CHECK_EQ(shape_size, permutation.size());
  for (wtf_size_t i = 0; i < shape_size; ++i) {
    permuted_array[i] = shape[permutation[i]];
  }

  return permuted_array;
}
}  // namespace

void OptionExpansionTransformer::Transform(MLNamedOperands& named_outputs) {
  auto* sorted_operators = GetOperatorsInTopologicalOrder(named_outputs);

  HeapHashSet<Member<const MLOperator>> graph_output_operators;
  for (auto& named_output : named_outputs) {
    auto* output_operand = named_output.second.Get();
    graph_output_operators.insert(output_operand->Operator());
  }

  for (auto& op : *sorted_operators) {
    MLOperand* original_operand = op->Outputs()[0].Get();
    MLOperand* updated_operand = original_operand;
    switch (op->Kind()) {
      case webnn::mojom::internal::Operation_Data::Operation_Tag::kConv2d: {
        switch (op->SubKind<blink_mojom::Conv2d::Kind>()) {
          case blink_mojom::Conv2d::Kind::kDirect: {
            updated_operand = HandleConv2d<MLConv2dOptions>(
                const_cast<MLOperator*>(op.Get()));
            break;
          }
          case blink_mojom::Conv2d::Kind::kTransposed: {
            updated_operand = HandleConv2d<MLConvTranspose2dOptions>(
                const_cast<MLOperator*>(op.Get()));
            break;
          }
        }
        break;
      }

      default:
        break;
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
}

template <typename MLConv2dOptionsType>
MLOperand* OptionExpansionTransformer::HandleConv2d(MLOperator* conv2d) {
  const auto* options =
      static_cast<const MLConv2dOptionsType*>(conv2d->Options());
  CHECK(options);

  auto context_properties = graph_builder_->GetContext()->GetProperties();
  auto exception_state = GetExceptionState();

  // Compute input potential permutation.
  const std::optional<base::span<const uint32_t>> input_permutation =
      GetInputOperandPermutation(options->inputLayout().AsEnum(),
                                 context_properties);
  // Compute filter potential permutation.
  std::optional<std::array<uint32_t, 4>> filter_permutation;
  if constexpr (std::is_same<MLConv2dOptionsType, MLConv2dOptions>::value) {
    bool depthwise = IsDepthwiseConv2d(conv2d);
    filter_permutation =
        GetConv2DFilterPermutation(context_properties.input_operand_layout,
                                   depthwise, options->filterLayout());

  } else if constexpr (std::is_same<MLConv2dOptionsType,
                                    MLConvTranspose2dOptions>::value) {
    filter_permutation = GetConvTranspose2DFilterPermutation(
        context_properties.input_operand_layout, options->filterLayout());
  } else {
    NOTREACHED();
  }

  // Insert input transpose if needed.
  if (input_permutation) {
    auto* conv2d_input_operand = conv2d->Inputs()[0].Get();

    Disconnect(conv2d_input_operand, conv2d, 0);

    MLTransposeOptions* transpose_options = MLTransposeOptions::Create();
    transpose_options->setPermutation(Vector<uint32_t>(*input_permutation));
    transpose_options->setLabel(options->label());
    auto* transpose_operand = graph_builder_->transpose(
        conv2d_input_operand, transpose_options, exception_state);
    Connect(transpose_operand, conv2d, 0);
    // update conv2d output shape, but cannot directly update the operand
    // because it's visiable for JS. We should create a new MLOperand instead.
    MLOperand* output_operand = conv2d->Outputs()[0];
    auto new_output_shape =
        PermuteShape(output_operand->Shape(), *input_permutation);
    auto* new_output_operand =
        CloneResetShape(output_operand, new_output_shape);
    ReplaceOperand(output_operand, new_output_operand);
  }

  // Insert filter transpose if needed.
  if (filter_permutation) {
    auto* filter_operand = conv2d->Inputs()[1].Get();
    Disconnect(filter_operand, conv2d, 1);
    MLTransposeOptions* transpose_options = MLTransposeOptions::Create();
    transpose_options->setPermutation(Vector<uint32_t>(*filter_permutation));
    transpose_options->setLabel(options->label());

    auto* transpose_operand = graph_builder_->transpose(
        filter_operand, transpose_options, exception_state);

    Connect(transpose_operand, conv2d, 1);
  }

  const std::optional<base::span<const uint32_t>> output_permutation =
      GetOutputOperandPermutation(options->inputLayout().AsEnum(),
                                  context_properties);

  // Insert output transpose if needed.
  auto* conv2d_output_operand = conv2d->Outputs()[0].Get();
  if (output_permutation) {
    HeapVector<std::pair<MLOperator*, int>> conv_output_ops_to_update;
    auto conv_output_ops = conv2d->Outputs()[0]->DependentOperators();
    for (auto& conv_output_op : conv_output_ops) {
      MLOperator* output_op = const_cast<MLOperator*>(conv_output_op.Get());
      int disconnect_index = Disconnect(conv2d, 0, output_op);
      conv_output_ops_to_update.push_back(
          std::pair{output_op, disconnect_index});
    }

    MLTransposeOptions* transpose_options = MLTransposeOptions::Create();
    transpose_options->setPermutation(Vector<uint32_t>(*output_permutation));
    transpose_options->setLabel(options->label());

    auto* transpose_operand = graph_builder_->transpose(
        conv2d->Outputs()[0], transpose_options, exception_state);

    for (auto& [output_op, index] : conv_output_ops_to_update) {
      Connect(transpose_operand, output_op, index);
    }
    conv2d_output_operand = transpose_operand;
  }
  return conv2d_output_operand;
}

template MLOperand* OptionExpansionTransformer::HandleConv2d<MLConv2dOptions>(
    MLOperator* conv2d);

template MLOperand* OptionExpansionTransformer::HandleConv2d<
    MLConvTranspose2dOptions>(MLOperator* conv2d);

}  // namespace blink
