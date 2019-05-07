#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

using std::vector;
using std::pair;
using std::string;

namespace nvinfer1
{
namespace plugin
{
static const vector<string> VOC_CLASSES{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
static const auto VOC_CLS_SIZE = VOC_CLASSES.size();

struct GTBBox
{
    float xmin, ymin, xmax, ymax;
    int label;
    bool matched;
    bool difficult;
};

void readAnnotation(const std::string& xmlFile, bool isPost2008,
                    float scaledImgH, float scaledImgW,
                    std::vector<GTBBox>& gt, std::vector<int>& numGT);

void evaluateDetection(std::vector<std::vector<std::pair<float, int>>>& tp,
                       std::vector<std::vector<std::pair<float, int>>>& fp,
                       std::vector<int>& totalGT,
                       std::vector<std::vector<float>>& precision,
                       std::vector<std::vector<float>>& recall,
                       std::vector<float>& AP,
                       float& mAP);

template <typename T>
class DetectionDecoder
{
public:
    virtual int getLabel(T* det, int detID) const = 0;
    virtual float getConfidence(T* det, int detID) const = 0;
    virtual float getXmin(T* det, int detID, int inputW) const = 0;
    virtual float getXmax(T* det, int detID, int inputW) const = 0;
    virtual float getYmin(T* det, int detID, int inputH) const = 0;
    virtual float getYmax(T* det, int detID, int inputH) const = 0;
};

template <typename T>
void updateAccuracyCounts(const float overlapThresh,
                          const int numDet,
                          const int inputH,
                          const int inputW,
                          T* detections,
                          DetectionDecoder<T>* decoder,
                          vector<GTBBox>& gt,
                          vector<vector<pair<float, int>>>& tp,
                          vector<vector<pair<float, int>>>& fp,
                          vector<int>& totalGT)
{
    assert(tp.size() == VOC_CLS_SIZE);
    assert(fp.size() == VOC_CLS_SIZE);
    assert(totalGT.size() == VOC_CLS_SIZE);

    for (int j = 0; j < numDet; ++j)
    {
        auto label = decoder->getLabel(detections, j);
        auto conf = decoder->getConfidence(detections, j);
        auto xmin = decoder->getXmin(detections, j, inputW),
             ymin = decoder->getYmin(detections, j, inputH),
             xmax = decoder->getXmax(detections, j, inputW),
             ymax = decoder->getYmax(detections, j, inputH);
        pair<float, int> tmpPair(conf, 0);
        tp[label].push_back(tmpPair);
        fp[label].push_back(tmpPair);

        float overlapMax = 0.0f;
        int overlapMaxInd = -1;
        for (unsigned k = 0; k < gt.size(); ++k)
        {
            if (gt[k].label == label)
            {
                auto ow = std::min(xmax, gt[k].xmax) - std::max(xmin, gt[k].xmin) + 1,
                     oh = std::min(ymax, gt[k].ymax) - std::max(ymin, gt[k].ymin) + 1;
                if (ow > 0 && oh > 0)
                {
                    auto ua = (xmax - xmin + 1) * (ymax - ymin + 1) + (gt[k].xmax - gt[k].xmin + 1) * (gt[k].ymax - gt[k].ymin + 1) - ow * oh;
                    auto overlap = ow * oh / ua;
                    if (overlap > overlapMax)
                    {
                        overlapMax = overlap;
                        overlapMaxInd = k;
                    } //if:overlap
                }     //if:ow,oh
            }         //if:gt[k].label
        }             //for:k
        // no corresponding GT was found (e.g. random false positive)
        if (overlapMaxInd == -1)
        {
            fp[label].back().second = 1;
        }
        // only score non-difficult samples
        else if (!gt[overlapMaxInd].difficult)
        {
            // if it's a match and the gt object hasn't already been detected
            if (overlapMax > overlapThresh && gt[overlapMaxInd].matched != 1)
            {
                gt[overlapMaxInd].matched = 1;
                tp[label].back().second = 1;
            }
            else
            {
                fp[label].back().second = 1;
            }
        } //else if:not difficult
    }     //for:j
    }

} // namespace plugin
} // namespace nvinfer1
