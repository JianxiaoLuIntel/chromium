#include "third_party/blink/renderer/modules/ml/webnn/optimizer/options_resolve_transformer.h"

#include "services/webnn/public/cpp/graph_validation_utils.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_transpose_2d_options.h"

namespace blink_mojom = webnn::mojom::blink;

namespace blink::webnn_optimizer {

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

void OptionsResolveTransformer::TransformInternal() {
  auto sorted = graph_->TopologicalSort();
  for (auto node : sorted) {
    if (node->op_kind() == OpKind::kConv2d) {
      auto* conv2d_node = static_cast<Conv2dNode*>(node);
      switch (conv2d_node->kind) {
        case webnn::mojom::Conv2d_Kind::kDirect:
          ResolveConv2d<MLConv2dOptions>(conv2d_node);
          break;
        case webnn::mojom::Conv2d_Kind::kTransposed:
          ResolveConv2d<MLConvTranspose2dOptions>(conv2d_node);
          break;
        default:
          NOTREACHED() << "Invalid Conv2d kind";
      }
    }
  }
}

bool IsDepthwiseConv2d(const MLConv2dOptions* options,
                       std::vector<uint32_t> input_shape,
                       std::vector<uint32_t> output_shape) {
  CHECK(options);
  CHECK_EQ(input_shape.size(), 4u);
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

template <typename MLConv2dOptionsType>
void OptionsResolveTransformer::ResolveConv2d(Conv2dNode* conv2d_nodes) {
  const auto context_properties = graph_->GetContextProperties();
  const auto* options = static_cast<const MLConv2dOptionsType*>(
      conv2d_nodes->GetMLOperatorOptions().Get());
  CHECK(options);
  const auto conv2d_output_desc = conv2d_nodes->GetOperandDescriptors()[0];
  const auto conv2d_input_desc =
      conv2d_nodes->GetInputNodes()[0]->GetOperandDescriptors()[0];
  const auto conv2d_filter_desc =
      conv2d_nodes->GetInputNodes()[1]->GetOperandDescriptors()[0];

  const std::optional<base::span<const uint32_t>> input_permutation =
      GetInputOperandPermutation(options->inputLayout().AsEnum(),
                                 *context_properties);
  // if need insert transpose node at input
  if (input_permutation.has_value()) {
    TransposeNode* transpose = MakeGarbageCollected<TransposeNode>();
    transpose->permutation = Vector<uint32_t>(*input_permutation);
    auto transpose_output_desc = ::webnn::OperandDescriptor::Create(
        *context_properties, conv2d_input_desc.data_type(),
        PermuteShape(conv2d_input_desc.shape(), *input_permutation),
        conv2d_nodes->GetLabel().Utf8());
    transpose->SetOperandDescriptors({transpose_output_desc.value()});
    auto origin_edge = conv2d_nodes->GetInputEdges()[0];
    auto* origin_from_node = origin_edge->FromNode();
    auto origin_from_idx = origin_edge->FromIndex();

    Edge::Disconnect(origin_from_node, origin_from_idx, conv2d_nodes, 0);
    Edge::Connect(origin_from_node, origin_from_idx, transpose, 0);
    Edge::Connect(transpose, 0, conv2d_nodes, 0);

    // Update output operand shape
    auto new_conv2d_output_desc = ::webnn::OperandDescriptor::Create(
        *context_properties, conv2d_output_desc.data_type(),
        PermuteShape(conv2d_output_desc.shape(), *input_permutation),
        conv2d_nodes->GetLabel().Utf8());

    conv2d_nodes->SetOperandDescriptors({new_conv2d_output_desc.value()});
  }

  std::optional<std::array<uint32_t, 4>> filter_permutation;
  if constexpr (std::is_same<MLConv2dOptionsType, MLConv2dOptions>::value) {
    bool depthwise = IsDepthwiseConv2d(
        options,
        conv2d_nodes->GetInputNodes()[0]->GetOperandDescriptors()[0].shape(),
        conv2d_output_desc.shape());

    filter_permutation =
        GetConv2DFilterPermutation(context_properties->input_operand_layout,
                                   depthwise, options->filterLayout());

  } else if constexpr (std::is_same<MLConv2dOptionsType,
                                    MLConvTranspose2dOptions>::value) {
    filter_permutation = GetConvTranspose2DFilterPermutation(
        context_properties->input_operand_layout, options->filterLayout());
  } else {
    NOTREACHED();
  }

  // If need insert transpose node at filter
  if (filter_permutation.has_value()) {
    TransposeNode* transpose = MakeGarbageCollected<TransposeNode>();
    transpose->permutation = Vector<uint32_t>(filter_permutation.value());
    auto transpose_output_desc = ::webnn::OperandDescriptor::Create(
        *context_properties, conv2d_filter_desc.data_type(),
        PermuteShape(conv2d_filter_desc.shape(), *filter_permutation),
        conv2d_nodes->GetLabel().Utf8());
    transpose->SetOperandDescriptors({transpose_output_desc.value()});

    auto origin_edge = conv2d_nodes->GetInputEdges()[1];
    auto* origin_from_node = origin_edge->FromNode();
    auto origin_from_idx = origin_edge->FromIndex();

    Edge::Disconnect(origin_from_node, origin_from_idx, conv2d_nodes, 1);
    Edge::Connect(origin_from_node, origin_from_idx, transpose, 0);
    Edge::Connect(transpose, 0, conv2d_nodes, 1);
  }

  // If need insert transpose node at output
  const std::optional<base::span<const uint32_t>> output_permutation =
      GetOutputOperandPermutation(options->inputLayout().AsEnum(),
                                  *context_properties);

  if (output_permutation.has_value()) {
    TransposeNode* transpose = MakeGarbageCollected<TransposeNode>();
    transpose->permutation = Vector<uint32_t>(*output_permutation);
    auto transpose_output_desc = ::webnn::OperandDescriptor::Create(
        *context_properties, conv2d_output_desc.data_type(),
        PermuteShape(conv2d_output_desc.shape(), *output_permutation),
        conv2d_nodes->GetLabel().Utf8());
    transpose->SetOperandDescriptors({transpose_output_desc.value()});

    auto outputs = conv2d_nodes->GetOutputPorts()[0];

    for (auto edge : outputs) {
      auto* origin_to_node = edge->ToNode();
      auto origin_to_idx = edge->ToIndex();
      Edge::Disconnect(conv2d_nodes, 0, origin_to_node, origin_to_idx);
      Edge::Connect(transpose, 0, origin_to_node, origin_to_idx);
    }
    Edge::Connect(conv2d_nodes, 0, transpose, 0);
  }
}

template void OptionsResolveTransformer::ResolveConv2d<MLConv2dOptions>(
    Conv2dNode* conv2d_nodes);

template void OptionsResolveTransformer::ResolveConv2d<
    MLConvTranspose2dOptions>(Conv2dNode* conv2d_nodes);

}  // namespace blink::webnn_optimizer
