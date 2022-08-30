#include "tut/constraintContext.h"
#include "tut/safety/tutRNG.h"
#include "tutTestRunner.h"

using namespace tut;
using namespace nvinfer1;

namespace
{

struct TestCase
{
    Coords dataDims;
    float epsilon;
    int32_t groups;
};

std::vector<TestCase> const fastConfigTestCases = {
    // clang-format off
    {{1,  6,  8,  8},    1e-5F,  3}
    // clang-format on
};

std::vector<TestCase> testCases = {
    // clang-format off
    {{2, 16,  8,  8},    1e-5F,   4},
    {{2, 16,  8,  8},    1e-5F,  16}
    // clang-format on
};

template <typename PtrType>
struct PluginDeletor
{
public:
    void operator()(PtrType ptr)
    {
        ptr->destroy();
    }
};
using PluginPtr = std::unique_ptr<IPluginV2, PluginDeletor<IPluginV2*> >;

struct PluginGroupNormTest : TestRunner
{
public:
    int32_t GetSubCount() const override
    {
        return mTestCases.size();
    }

    void SetupTestCases() override
    {
        mTestCases = concatTestsIfNotFastConfig(fastConfigTestCases, testCases);
    }

    void testApi(TestFormat const& format)
    {
        disableImplicitBatch("pluginv2Dynamic doesn't support implict batch");
        fullDimsLayerBuilder = true;

        auto creator = getPluginRegistry()->getPluginCreator("GroupNormalizationPlugin", "1");
        RETURN_IF_FALSE(creator);

        for (int32_t i = 0, n = signedSize(mTestCases); i < n; ++i)
        {
            subIndex = i;
            auto const& test = mTestCases[i];
            if (filter(i, volume(test.dataDims)))
            {
                continue;
            }

            setup(test, format, seed);

            std::vector<nvinfer1::PluginField> fc;
            fc.emplace_back("eps", &test.epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
            fc.emplace_back("num_groups", &test.groups, nvinfer1::PluginFieldType::kINT32, 1);

            PluginFieldCollection const pfc{static_cast<int32_t>(fc.size()), fc.data()};
            PluginPtr grpNormPlugin{
                creator->createPlugin("GroupNormalizationPlugin", &pfc), PluginDeletor<IPluginV2*>()};
            RETURN_IF_FALSE(grpNormPlugin.get());

            bool const correct = runApiTest(0, format, [&](INetworkDefinition& n, IBuilderConfig& b) {
                // enable CUDNN tactic source because this plugin used cudnn.
                constexpr auto kCUDNN_BIT{1U << static_cast<uint32_t>(TacticSource::kCUDNN)};
                auto ts = static_cast<uint32_t>(b.getTacticSources());
                ts = ts | kCUDNN_BIT;
                b.setTacticSources(static_cast<TacticSources>(ts));
                b.setPreviewFeature(PreviewFeature::k0805_DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE, false);

                // add plugin.
                Coords const scaleDims{test.dataDims[1]};
                ITensor* inputs[] = {
                    n.addInput("data", getDataType(format.in(0)), dimsOf(test.dataDims)),
                    n.addInput("scales", getDataType(format.in(0)), dimsOf(scaleDims)),
                    n.addInput("bias", getDataType(format.in(0)), dimsOf(scaleDims)),
                };
                auto layer = n.addPluginV2(inputs, sizeof(inputs) / sizeof(inputs[0]), *grpNormPlugin.get());

                markOutput(n, *layer->getOutput(0), "output");
                ITensor* out = layer->getOutput(0);
                out->setType(getDataType(format.out(0)));
            });
            EXPECT_TRUE(correct);
            teardown();
        }
    }

private:
    void setup(TestCase const& testCase, TestFormat const& format, int32_t seed)
    {
        tolerance = 1e-2F;
        SimpleRNG rng(-5, 5, 4, seed);
        FloatRNG scalerng(0.9F, 1.1F, 0, seed);
        FloatRNG biasrng(0.F, 0.2F, 0, seed);
        Coords const inputDims = testCase.dataDims;
        EXPECTS(inputDims.nbDims == 4);

        Coords const inputOffset = createOffset(inputDims.size, format.in(0), rng, RandomOffsetType::kNONE);
        mInputs.emplace_back(TensorBuilder(gpuAllocator(), "data", inputDims, inputOffset, format.in(0))
                                 .tensorMode(TensorMode::kINITIALIZED)
                                 .rng(rng)
                                 .build());
        Coords const scaleDims{inputDims[1]};
        Coords const scaleOffset = createOffset(scaleDims.size, format.in(0), rng, RandomOffsetType::kNONE);
        mInputs.emplace_back(TensorBuilder(gpuAllocator(), "scales", scaleDims, scaleOffset, format.in(0))
                                 .tensorMode(TensorMode::kINITIALIZED)
                                 .rng(scalerng)
                                 .build());
        mInputs.emplace_back(TensorBuilder(gpuAllocator(), "bias", scaleDims, scaleOffset, format.in(0))
                                 .tensorMode(TensorMode::kINITIALIZED)
                                 .rng(biasrng)
                                 .build());

        // reshape (N, C, H, W) -> (N, group, -1).
        Permutation identityPerm{};
        std::iota(identityPerm.order, identityPerm.order + inputDims.nbDims, 0);
        Coords foldDims{inputDims[0], testCase.groups, volume(inputDims) / inputDims[0] / testCase.groups};
        Box intermediateBox(foldDims, BoxType::kFLOAT);
        ShuffleParameters const foldParams{identityPerm, dimsOf(foldDims), identityPerm, true};
        mRef.refShuffle(mInputs[0]->toBox().get<float const>(), intermediateBox.get<float>(), foldParams, true);

        // run instanceNorm.
        Box instNormBox(foldDims, BoxType::kFLOAT);
        std::vector<float> scales(testCase.groups, 1);
        std::vector<float> bias(testCase.groups, 0);
        mRef.refInstanceNorm(intermediateBox.get<float const>(), instNormBox.get<float>(), scales.data(), bias.data(),
            false, 0, testCase.epsilon);

        // reshape (N, group, -1) -> (N, C, H, W).
        Box unfoldBox(inputDims, BoxType::kFLOAT);
        ShuffleParameters const unfoldShape{identityPerm, dimsOf(inputDims), identityPerm, true};
        mRef.refShuffle(instNormBox.get<float const>(), unfoldBox.get<float>(), unfoldShape, true);

        // scale by (1, C, 1, 1).
        Coords const elementWiseDims{1, inputDims[1], 1, 1};
        ShuffleParameters const unsqueezeParam{identityPerm, dimsOf(elementWiseDims), identityPerm, true};
        Box scalesBox(elementWiseDims, BoxType::kFLOAT);
        mRef.refShuffle(mInputs[1]->toBox().get<float const>(), scalesBox.get<float>(), unsqueezeParam, true);
        Box afterScaleBox(inputDims, BoxType::kFLOAT);
        mRef.refElementWise(unfoldBox.get<float const>(), scalesBox.get<float const>(), afterScaleBox.get<float>(),
            ElementWiseParameters{ElementWiseOperation::kPROD});

        // bias by (1, C, 1, 1).
        Box biasBox(elementWiseDims, BoxType::kFLOAT);
        mRef.refShuffle(mInputs[2]->toBox().get<float const>(), biasBox.get<float>(), unsqueezeParam, true);
        Box refBox(inputDims, BoxType::kFLOAT);
        mRef.refElementWise(afterScaleBox.get<float const>(), biasBox.get<float const>(), refBox.get<float>(),
            ElementWiseParameters{ElementWiseOperation::kSUM});

        mRefOutputs.emplace_back(TensorBuilder(gpuAllocator(), "output", inputDims, inputOffset, format.out(0))
                                     .tensorMode(TensorMode::kINITIALIZED_BOX)
                                     .rng(rng)
                                     .initializer(refBox)
                                     .build());
    }

    std::vector<TestCase> mTestCases;
};

struct tutApiPluginGroupNormTest : PluginGroupNormTest
{
};

TEST_GPU(tutApiPluginGroupNormTest, FP32)
{
    testApi(FP32_SCALAR);
}

} // namespace
