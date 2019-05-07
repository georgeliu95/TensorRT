#include "vocEvaluate.h"
#include <cmath>
#include <iostream>

using std::vector;
using std::string;
using std::ifstream;
using std::getline;
using std::stoi;
using std::pair;

namespace nvinfer1
{
namespace plugin
{
void readAnnotation(const string& xmlFile, const bool isPost2008,
                    float scaledImgH, float scaledImgW,
                    vector<GTBBox>& gt, vector<int>& numGT)
{
    ifstream file(xmlFile);
    if (file.fail())
    {
        printf("Unable to find %s.\n", xmlFile.c_str());
    }
    assert(!file.fail());

    string line, lineStripped;
    float scaleWidth = 1.0f, scaleHeight = 1.0f;
    auto stripNextLineOf = [](ifstream& f) -> string {
        string line;
        getline(f, line);
        auto first = line.find_first_not_of('\t');
        return line.substr(first, line.length() - first);
    };
    auto getBetwAngles = [](ifstream& f) -> string {
        string line;
        getline(f, line);
        auto first = line.find_first_of('>');
        auto last = line.find_last_of('<');
        return line.substr(first + 1, last - first - 1);
    };
    while (getline(file, line))
    {
        auto first = line.find_first_not_of('\t');
        if (first == line.length() - 1)
        {
            break;
        }

        lineStripped = line.substr(first, line.length() - first);
        if (lineStripped == "<size>")
        {
            // width
            scaleWidth = scaledImgW / stoi(getBetwAngles(file));

            // height
            scaleHeight = scaledImgH / stoi(getBetwAngles(file));
        }
        else if (lineStripped == "<object>")
        {
            GTBBox tmpGT{};
            tmpGT.label = find(VOC_CLASSES.begin(), VOC_CLASSES.end(), getBetwAngles(file)) - VOC_CLASSES.begin();
            tmpGT.matched = 0;
            if (!isPost2008)
            {
                // skip pose, truncated
                getline(file, line);
                getline(file, line);

                // difficult
                tmpGT.difficult = stoi(getBetwAngles(file));
            }
            else
            {
                tmpGT.difficult = 0;
            }

            // skip bndbox line
            while (stripNextLineOf(file) == "<part>")
            {
                for (int i = 0; i < 8; ++i)
                {
                    getline(file, line);
                }
            }

            // xmin, ymin, xmax, ymax
            tmpGT.xmin = std::floor(scaleWidth * stoi(getBetwAngles(file)));
            tmpGT.ymin = std::floor(scaleHeight * stoi(getBetwAngles(file)));
            tmpGT.xmax = std::floor(scaleWidth * stoi(getBetwAngles(file)));
            tmpGT.ymax = std::floor(scaleHeight * stoi(getBetwAngles(file)));

            gt.push_back(tmpGT);
            if (!tmpGT.difficult)
            {
                numGT[tmpGT.label]++;
            }
        }
        else
        {
            continue;
        }
    }

    }

void evaluateDetection(vector<vector<pair<float, int>>>& tp,
                       vector<vector<pair<float, int>>>& fp,
                       vector<int>& totalGT,
                       vector<vector<float>>& precision,
                       vector<vector<float>>& recall,
                       vector<float>& AP,
                       float& mAP)
{
    assert(tp.size() == VOC_CLS_SIZE);
    assert(fp.size() == VOC_CLS_SIZE);
    assert(totalGT.size() == VOC_CLS_SIZE);
    assert(precision.size() == VOC_CLS_SIZE);
    assert(recall.size() == VOC_CLS_SIZE);
    assert(AP.size() == VOC_CLS_SIZE);

    mAP = 0.0f;
    int cls = 0;
    for (unsigned i = 0; i < VOC_CLS_SIZE; ++i)
    {
        if (!tp[i].empty())
        {
            auto comp = [](const pair<float, int>& i, const pair<float, int>& j) -> bool {
                return i.first > j.first;
            };
            std::sort(tp[i].begin(), tp[i].end(), comp);
            std::sort(fp[i].begin(), fp[i].end(), comp);

            for (unsigned j = 1; j < tp[i].size(); ++j)
            {
                tp[i][j].second += tp[i][j - 1].second;
                fp[i][j].second += fp[i][j - 1].second;
            }

            for (unsigned int j = 0; j < tp[i].size(); ++j)
            {
                if (!totalGT[i])
                {
                    recall[i].push_back(0.0f);
                }
                else
                {
                    recall[i].push_back(1.0f * tp[i][j].second / totalGT[i]);
                }
                precision[i].push_back(1.0f * tp[i][j].second / (tp[i][j].second + fp[i][j].second));
            }

            // VOCDevkit implementation
            float one = 1.0f;
            float zero = 0.0f;
            recall[i].insert(recall[i].begin(), zero);
            precision[i].insert(precision[i].begin(), zero);

            recall[i].push_back(one);
            precision[i].push_back(zero);

            // interpolated AP
            for (int j = precision[i].size() - 2; j >= 0; --j)
            {
                if (precision[i][j + 1] > precision[i][j])
                {
                    precision[i][j] = precision[i][j + 1];
                }
            }

            vector<int> ids;
            for (unsigned int j = 0; j < recall[i].size() - 2; ++j)
            {
                // skip the 0 inserted in the front and the 1 inserted in the end
                if (std::fabs(recall[i][j] - recall[i][j + 1]) > 1e-10)
                {
                    ids.push_back(j + 1);
                }
            }

            AP[i] = 0.0f;
            auto recLen = recall[i].size(), precLen = precision[i].size();
            for (auto id : ids)
            {
                //outFile << recall[i][id] << ", " << precision[i][id] << "\n";
                recall[i].push_back(recall[i][id]);
                precision[i].push_back(precision[i][id]);
                // integral
                AP[i] += (recall[i][id] - recall[i][id - 1]) * precision[i][id];
            }
            recall[i].erase(recall[i].begin(), recall[i].begin() + recLen);
            precision[i].erase(precision[i].begin(), precision[i].begin() + precLen);

            mAP += AP[i];
            cls++;
        } //if:tp[i].size()
    }     //for:i
    mAP /= (float) cls;
}

} // namespace plugin
} // namespace nvinfer1
