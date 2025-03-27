#include "third_party/blink/renderer/modules/ml/webnn/optimizer/ml_operator_converter.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_arg_min_max_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_batch_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_transpose_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_cumulative_sum_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_elu_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gather_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gru_cell_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gru_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_hard_sigmoid_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_input_operand_layout.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_instance_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_layer_normalization_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_leaky_relu_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_linear_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_lstm_cell_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_lstm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operator_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pad_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_recurrent_network_activation.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_reduce_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_resample_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_scatter_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_slice_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_split_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_transpose_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_triangular_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink::webnn_optimizer {

Node* ConvertMLOperatorToNode(const MLOperator* op) {
  auto tag = op->Kind();
  Node* ret = nullptr;
  switch (tag) {
    case webnn::mojom::blink::Operation::Tag::kArgMinMax: {
      const auto* ml_argminmax = static_cast<const MLArgMinMaxOperator*>(op);
      auto* node = MakeGarbageCollected<ArgMinMaxNode>();
      const auto* options =
          static_cast<const blink::MLArgMinMaxOptions*>(op->Options());
      CHECK(options);
      node->axis = ml_argminmax->Axis();
      node->keep_dimensions = options->keepDimensions();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kBatchNormalization: {
      auto* node = MakeGarbageCollected<BatchNormalizationNode>();
      const auto* options =
          static_cast<const blink::MLBatchNormalizationOptions*>(op->Options());
      node->axis = options->axis();
      node->epsilon = options->epsilon();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kClamp: {
      auto* node = MakeGarbageCollected<ClampNode>();
      const auto* options =
          static_cast<const blink::MLClampOptions*>(op->Options());
      node->min_value =
          options->getMinValueOr(-std::numeric_limits<float>::infinity());
      node->max_value =
          options->getMaxValueOr(+std::numeric_limits<float>::infinity());
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kConcat: {
      auto* node = MakeGarbageCollected<ConcatNode>();
      const auto* ml_concat = static_cast<const MLConcatOperator*>(op);
      const auto input_num = op->Inputs().size();
      node->SetInputNum(input_num);
      node->axis = ml_concat->Axis();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kConv2d: {
      auto* node = MakeGarbageCollected<Conv2dNode>();
      node->kind = static_cast<webnn::mojom::Conv2d_Kind>(op->Kind());
      if (node->kind == webnn::mojom::Conv2d_Kind::kDirect) {
        const auto* options =
            static_cast<const blink::MLConv2dOptions*>(op->Options());

        auto strides = options->getStridesOr({1, 1});
        CHECK_EQ(strides.size(), 2u);
        node->strides =
            webnn::mojom::blink::Size2d::New(strides[0], strides[1]);

        auto dilations = options->getDilationsOr({1, 1});
        CHECK_EQ(dilations.size(), 2u);
        node->dilations =
            webnn::mojom::blink::Size2d::New(dilations[0], dilations[1]);
        node->groups = options->groups();

        auto ml_padding = options->getPaddingOr({0, 0, 0, 0});
        CHECK_EQ(ml_padding.size(), 4u);
        node->padding = webnn::mojom::blink::Padding2d::New(
            webnn::mojom::blink::Size2d::New(ml_padding[0], ml_padding[2]),
            webnn::mojom::blink::Size2d::New(ml_padding[1], ml_padding[3]));
      } else {
        DCHECK_EQ(node->kind, webnn::mojom::Conv2d_Kind::kTransposed);
        const auto* options =
            static_cast<const blink::MLConvTranspose2dOptions*>(op->Options());
        auto strides = options->getStridesOr({1, 1});
        CHECK_EQ(strides.size(), 2u);
        node->strides =
            webnn::mojom::blink::Size2d::New(strides[0], strides[1]);

        auto dilations = options->getDilationsOr({1, 1});
        CHECK_EQ(dilations.size(), 2u);
        node->dilations =
            webnn::mojom::blink::Size2d::New(dilations[0], dilations[1]);
        node->groups = options->groups();

        auto ml_padding = options->getPaddingOr({0, 0, 0, 0});
        CHECK_EQ(ml_padding.size(), 4u);
        node->padding = webnn::mojom::blink::Padding2d::New(
            webnn::mojom::blink::Size2d::New(ml_padding[0], ml_padding[2]),
            webnn::mojom::blink::Size2d::New(ml_padding[1], ml_padding[3]));
      }
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kCumulativeSum: {
      const auto* cumulative_sum =
          static_cast<const MLCumulativeSumOperator*>(op);
      auto* node = MakeGarbageCollected<CumulativeSumNode>();
      const auto* options =
          static_cast<const blink::MLCumulativeSumOptions*>(op->Options());
      node->axis = cumulative_sum->Axis();
      node->exclusive = options->exclusive();
      node->reversed = options->reversed();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kDequantizeLinear: {
      auto* node = MakeGarbageCollected<DequantizeLinearNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kElementWiseBinary: {
      auto* node = MakeGarbageCollected<ElementWiseBinaryNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kElementWiseUnary: {
      auto* node = MakeGarbageCollected<ElementWiseUnaryNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kExpand: {
      auto* node = MakeGarbageCollected<ExpandNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kElu: {
      auto* node = MakeGarbageCollected<EluNode>();
      const auto* options =
          static_cast<const blink::MLEluOptions*>(op->Options());
      node->alpha = options->alpha();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGather: {
      auto* node = MakeGarbageCollected<GatherNode>();
      const auto* options =
          static_cast<const blink::MLGatherOptions*>(op->Options());
      node->axis = options->axis();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGatherElements: {
      auto* node = MakeGarbageCollected<GatherElementsNode>();
      const auto* options =
          static_cast<const blink::MLGatherOptions*>(op->Options());
      node->axis = options->axis();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGatherNd: {
      auto* node = MakeGarbageCollected<GatherNDNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGelu: {
      auto* node = MakeGarbageCollected<GeluNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGemm: {
      auto* node = MakeGarbageCollected<GemmNode>();
      const auto* options =
          static_cast<const blink::MLGemmOptions*>(op->Options());
      node->alpha = options->alpha();
      node->beta = options->beta();
      node->a_transpose = options->aTranspose();
      node->b_transpose = options->bTranspose();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGru: {
      auto* node = MakeGarbageCollected<GruNode>();
      const auto output_num = op->Outputs().size();
      node->SetOutputPortNum(output_num);
      const auto* ml_gru = static_cast<const MLGruOperator*>(op);
      const auto* options =
          static_cast<const blink::MLGruOptions*>(op->Options());

      node->steps = ml_gru->steps();
      node->hidden_size = ml_gru->hidden_size();
      node->reset_after = options->resetAfter();
      node->return_sequence = options->returnSequence();

      switch (options->direction().AsEnum()) {
        case blink::V8MLRecurrentNetworkDirection::Enum::kForward:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kForward;
          break;
        case blink::V8MLRecurrentNetworkDirection::Enum::kBackward:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kBackward;
          break;
        case blink::V8MLRecurrentNetworkDirection::Enum::kBoth:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kBoth;
          break;
        default:
          NOTREACHED() << "Invalid direction";
      }

      switch (options->layout().AsEnum()) {
        case blink::V8MLGruWeightLayout::Enum::kZrn:
          node->layout = webnn::mojom::blink::GruWeightLayout::kZrn;
          break;
        case blink::V8MLGruWeightLayout::Enum::kRzn:
          node->layout = webnn::mojom::blink::GruWeightLayout::kRzn;
          break;
        default:
          NOTREACHED() << "Invalid layout";
      }

      CHECK_EQ(options->activations().size(), 2u);

      for (const auto& activation : options->activations()) {
        switch (activation.AsEnum()) {
          case blink::V8MLRecurrentNetworkActivation::Enum::kRelu:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kRelu);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kSigmoid:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kSigmoid);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kTanh:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kTanh);
            break;
          default:
            NOTREACHED() << "Invalid activation";
        }
      }

      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kGruCell: {
      auto* node = MakeGarbageCollected<GruCellNode>();
      const auto* ml_gru_cell = static_cast<const MLGruCellOperator*>(op);
      const auto* options =
          static_cast<const blink::MLGruCellOptions*>(op->Options());
      node->hidden_size = ml_gru_cell->hidden_size();
      node->reset_after = options->resetAfter();
      switch (options->layout().AsEnum()) {
        case blink::V8MLGruWeightLayout::Enum::kZrn:
          node->layout = webnn::mojom::blink::GruWeightLayout::kZrn;
          break;
        case blink::V8MLGruWeightLayout::Enum::kRzn:
          node->layout = webnn::mojom::blink::GruWeightLayout::kRzn;
          break;
        default:
          NOTREACHED() << "Invalid layout";
      }
      CHECK_EQ(options->activations().size(), 2u);

      for (const auto& activation : options->activations()) {
        switch (activation.AsEnum()) {
          case blink::V8MLRecurrentNetworkActivation::Enum::kRelu:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kRelu);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kSigmoid:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kSigmoid);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kTanh:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kTanh);
            break;
          default:
            NOTREACHED() << "Invalid activation";
        }
      }
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kHardSigmoid: {
      auto* node = MakeGarbageCollected<HardSigmoidNode>();
      const auto* options =
          static_cast<const blink::MLHardSigmoidOptions*>(op->Options());
      node->alpha = options->alpha();
      node->beta = options->beta();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kHardSwish: {
      auto* node = MakeGarbageCollected<HardSwishNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kInstanceNormalization: {
      auto* node = MakeGarbageCollected<InstanceNormalizationNode>();
      const auto* options =
          static_cast<const blink::MLInstanceNormalizationOptions*>(
              op->Options());
      node->epsilon = options->epsilon();
      switch (options->layout().AsEnum()) {
        case blink::V8MLInputOperandLayout::Enum::kNchw:
          node->layout =
              webnn::mojom::blink::InputOperandLayout::kChannelsFirst;
          break;
        case blink::V8MLInputOperandLayout::Enum::kNhwc:
          node->layout = webnn::mojom::blink::InputOperandLayout::kChannelsLast;
          break;
        default:
          NOTREACHED() << "Invalid layout";
      }

      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kLayerNormalization: {
      auto* node = MakeGarbageCollected<LayerNormalizationNode>();
      const auto* options =
          static_cast<const blink::MLLayerNormalizationOptions*>(op->Options());
      node->axes = options->axes();
      node->epsilon = options->epsilon();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kLeakyRelu: {
      auto* node = MakeGarbageCollected<LeakyReluNode>();
      const auto* options =
          static_cast<const blink::MLLeakyReluOptions*>(op->Options());
      node->alpha = options->alpha();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kLinear: {
      auto* node = MakeGarbageCollected<LinearNode>();
      const auto* options =
          static_cast<const blink::MLLinearOptions*>(op->Options());
      node->alpha = options->alpha();
      node->beta = options->beta();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kLstm: {
      auto* node = MakeGarbageCollected<LstmNode>();
      const auto output_num = op->Outputs().size();
      node->SetOutputPortNum(output_num);
      const auto* ml_lstm = static_cast<const MLLstmOperator*>(op);
      const auto* options =
          static_cast<const blink::MLLstmOptions*>(op->Options());
      node->steps = ml_lstm->steps();
      node->hidden_size = ml_lstm->hidden_size();
      node->return_sequence = options->returnSequence();
      switch (options->direction().AsEnum()) {
        case blink::V8MLRecurrentNetworkDirection::Enum::kForward:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kForward;
          break;
        case blink::V8MLRecurrentNetworkDirection::Enum::kBackward:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kBackward;
          break;
        case blink::V8MLRecurrentNetworkDirection::Enum::kBoth:
          node->direction =
              webnn::mojom::blink::RecurrentNetworkDirection::kBoth;
          break;
        default:
          NOTREACHED() << "Invalid direction";
      }

      switch (options->layout().AsEnum()) {
        case blink::V8MLLstmWeightLayout::Enum::kIfgo:
          node->layout = webnn::mojom::blink::LstmWeightLayout::kIfgo;
          break;
        case blink::V8MLLstmWeightLayout::Enum::kIofg:
          node->layout = webnn::mojom::blink::LstmWeightLayout::kIofg;
          break;
        default:
          NOTREACHED() << "Invalid layout";
      }

      CHECK_EQ(options->activations().size(), 3u);
      for (const auto& activation : options->activations()) {
        switch (activation.AsEnum()) {
          case blink::V8MLRecurrentNetworkActivation::Enum::kRelu:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kRelu);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kSigmoid:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kSigmoid);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kTanh:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kTanh);
            break;
          default:
            NOTREACHED() << "Invalid activation";
        }
      }
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kLstmCell: {
      auto* node = MakeGarbageCollected<LstmCellNode>();
      const auto output_num = op->Outputs().size();
      node->SetOutputPortNum(output_num);
      const auto* ml_lstmcell = static_cast<const MLLstmCellOperator*>(op);
      const auto* options =
          static_cast<const blink::MLLstmCellOptions*>(op->Options());
      node->hidden_size = ml_lstmcell->hidden_size();
      switch (options->layout().AsEnum()) {
        case blink::V8MLLstmWeightLayout::Enum::kIfgo:
          node->layout = webnn::mojom::blink::LstmWeightLayout::kIfgo;
          break;
        case blink::V8MLLstmWeightLayout::Enum::kIofg:
          node->layout = webnn::mojom::blink::LstmWeightLayout::kIofg;
          break;
        default:
          NOTREACHED() << "Invalid layout";
      }

      CHECK_EQ(options->activations().size(), 3u);
      for (const auto& activation : options->activations()) {
        switch (activation.AsEnum()) {
          case blink::V8MLRecurrentNetworkActivation::Enum::kRelu:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kRelu);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kSigmoid:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kSigmoid);
            break;
          case blink::V8MLRecurrentNetworkActivation::Enum::kTanh:
            node->activations.push_back(
                webnn::mojom::blink::RecurrentNetworkActivation::kTanh);
            break;
          default:
            NOTREACHED() << "Invalid activation";
        }
      }
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kMatmul: {
      auto* node = MakeGarbageCollected<MatmulNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kPad: {
      auto* node = MakeGarbageCollected<PadNode>();
      const auto* ml_pad = static_cast<const MLPadOperator*>(op);
      const auto* options =
          static_cast<const blink::MLPadOptions*>(op->Options());
      node->beginning_padding = ml_pad->BeginningPadding();
      node->ending_padding = ml_pad->EndingPadding();

      switch (options->mode().AsEnum()) {
        case V8MLPaddingMode::Enum::kConstant: {
          auto constant_padding = webnn::mojom::blink::ConstantPadding::New();
          constant_padding->value = options->value();
          node->mode = webnn::mojom::blink::PaddingMode::NewConstant(
              std::move(constant_padding));
          break;
        }
        case V8MLPaddingMode::Enum::kEdge:
          node->mode = webnn::mojom::blink::PaddingMode::NewEdge(
              webnn::mojom::blink::EdgePadding::New());
          break;
        case V8MLPaddingMode::Enum::kReflection:
          node->mode = webnn::mojom::blink::PaddingMode::NewReflection(
              webnn::mojom::blink::ReflectionPadding::New());
          break;
        case V8MLPaddingMode::Enum::kSymmetric:
          node->mode = webnn::mojom::blink::PaddingMode::NewSymmetric(
              webnn::mojom::blink::SymmetricPadding::New());
          break;
      }
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kPool2d: {
      auto* node = MakeGarbageCollected<Pool2dNode>();
      const auto* options =
          static_cast<const blink::MLPool2dOptions*>(op->Options());

      node->window_dimensions.height = options->windowDimensions()[0];
      node->window_dimensions.width = options->windowDimensions()[1];

      node->padding.beginning->height = options->padding()[0];
      node->padding.beginning->width = options->padding()[2];
      node->padding.ending->height = options->padding()[1];
      node->padding.ending->width = options->padding()[3];

      node->strides.height = options->strides()[0];
      node->strides.width = options->strides()[1];

      node->dilations.height = options->dilations()[0];
      node->dilations.width = options->dilations()[1];

      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kPrelu: {
      auto* node = MakeGarbageCollected<PreluNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kQuantizeLinear: {
      auto* node = MakeGarbageCollected<QuantizeLinearNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kReduce: {
      auto* node = MakeGarbageCollected<ReduceNode>();
      const auto* options =
          static_cast<const blink::MLReduceOptions*>(op->Options());
      node->axes = options->axes();
      node->keep_dimensions = options->keepDimensions();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kRelu: {
      auto* node = MakeGarbageCollected<ReluNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kResample2d: {
      auto* node = MakeGarbageCollected<Resample2dNode>();
      const auto* options =
          static_cast<const blink::MLResample2dOptions*>(op->Options());
      switch (options->mode().AsEnum()) {
        case blink::V8MLInterpolationMode::Enum::kNearestNeighbor:
          node->mode = webnn::mojom::blink::Resample2d::InterpolationMode::
              kNearestNeighbor;
          break;
        case blink::V8MLInterpolationMode::Enum::kLinear:
          node->mode =
              webnn::mojom::blink::Resample2d::InterpolationMode::kLinear;
          break;
      }
      node->scales = options->scales();
      node->axes = options->axes();

      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kReshape: {
      auto* node = MakeGarbageCollected<ReshapeNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kReverse: {
      auto* node = MakeGarbageCollected<ReverseNode>();
      const auto* ml_reverse = static_cast<const MLReverseOperator*>(op);
      node->axes = ml_reverse->Axes();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kScatterElements: {
      auto* node = MakeGarbageCollected<ScatterElementsNode>();
      const auto* options =
          static_cast<const blink::MLScatterOptions*>(op->Options());
      node->axis = options->axis();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kScatterNd: {
      auto* node = MakeGarbageCollected<ScatterNDNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSigmoid: {
      auto* node = MakeGarbageCollected<SigmoidNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSlice: {
      auto* node = MakeGarbageCollected<SliceNode>();
      const auto* ml_slice = static_cast<const MLSliceOperator*>(op);
      node->ranges.reserve(ml_slice->Starts().size());
      for (wtf_size_t i = 0; i < ml_slice->Starts().size(); i++) {
        node->ranges.emplace_back(ml_slice->Starts()[i], ml_slice->Sizes()[i],
                                  ml_slice->Strides()[i]);
      }
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSoftmax: {
      const auto* ml_softmax = static_cast<const MLSoftmaxOperator*>(op);
      auto* node = MakeGarbageCollected<SoftmaxNode>();
      node->axis = ml_softmax->Axis();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSoftplus: {
      auto* node = MakeGarbageCollected<SoftplusNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSoftsign: {
      auto* node = MakeGarbageCollected<SoftsignNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kSplit: {
      auto* node = MakeGarbageCollected<SplitNode>();
      const auto output_num = op->Outputs().size();
      node->SetOutputPortNum(output_num);
      const auto* options =
          static_cast<const blink::MLSplitOptions*>(op->Options());
      CHECK(options);
      if (options->hasAxis()) {
        node->axis = options->axis();
      }
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kTanh: {
      auto* node = MakeGarbageCollected<TanhNode>();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kTile: {
      auto* node = MakeGarbageCollected<TileNode>();
      const auto* ml_tile = static_cast<const MLTileOperator*>(op);
      node->repetitions = ml_tile->Repetitions();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kTranspose: {
      auto* node = MakeGarbageCollected<TransposeNode>();
      auto* options = static_cast<const MLTransposeOptions*>(op->Options());
      CHECK(options);
      wtf_size_t input_rank = op->Inputs()[0]->Rank();
      node->permutation =
          options->getPermutationOr(CreateDefaultPermutation(input_rank));
      CHECK_EQ(node->permutation.size(), input_rank);
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kTriangular: {
      auto* node = MakeGarbageCollected<TriangularNode>();
      const auto* options =
          static_cast<const blink::MLTriangularOptions*>(op->Options());
      node->upper = options->upper();
      node->diagonal = options->diagonal();
      ret = node;
      break;
    }
    case webnn::mojom::blink::Operation::Tag::kWhere: {
      auto* node = MakeGarbageCollected<WhereNode>();
      ret = node;
      break;
    }
    default:
      NOTIMPLEMENTED() << "Tag:" << static_cast<int>(tag)
                       << " not implemented yet";
  }

  CHECK(ret);
  ret->SetLabel(op->Options()->label());
  ret->SetOperands(op->Outputs());

  return ret;
}

}  // namespace blink::webnn_optimizer
