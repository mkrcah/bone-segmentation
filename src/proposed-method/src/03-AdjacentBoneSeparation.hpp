/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/






#pragma once


#include "GraphCut.hpp"
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include "Globals.hpp"


namespace BoneSeparation {


typedef GraphCutSegmentation<Dimension> GCSegm;


/** Label of an island */
typedef unsigned Label;

typedef pair<Label, unsigned> PairLabelSize;





/**
Info about one island.

This is a recursive structure:

    IslandStats
        info about the whole image,

    IslandStats.subIslands[x]
        info about mainIsland labelled x

    IslandStats.subIslands[x].subIslands[y]
        info about subIsland labelled y within x

*/
struct IslandStats {

    // number of pixels in the island
    unsigned count;

    // should this island be processed by the graph-cut separation?
    bool active;

    map<Label, IslandStats> subIslands;

    // constructor
    IslandStats() : count(0), active(false) {}

};






/**
Compute sizes of main islands and the corresponding subislands
*/
IslandStats countSizeOfIslands(
    UIntImagePtr mainIslands, UIntImagePtr subIslands
) {

    IslandStats stats;

    itk::ImageRegionIterator<UIntImage> itMain(
        mainIslands, mainIslands->GetLargestPossibleRegion());
    itk::ImageRegionIterator<UIntImage> itSub(
        subIslands, subIslands->GetLargestPossibleRegion());

    for (
        itMain.GoToBegin(), itSub.GoToBegin();
        !itMain.IsAtEnd();
        ++itMain, ++itSub
    ) {

        Label main  = itMain.Get();
        Label sub = itSub.Get();

        // number of pixels in the whole image
        stats.count ++;

        // skip pixels outside any main-island
        if (main == 0) {
            assert(sub == 0);
            continue;
        }

        // retrieve stats for the current main island
        IslandStats & islandStats = stats.subIslands[main];

        // increase number of pixels in this main island
        islandStats.count++;

        if (sub == 0)
            continue;

        // increase number of pixels in the corresponding sub-island
        islandStats.subIslands[sub].count++;
    }


    return stats;
}






/**
Distinguish which main islands should be processed,
i.e. in which main-islands the erosion caused separation
of the islands into more (nontrivial) sub-islands.
*/
void markIslandsToProcess(IslandStats & stats) {

    // each main label
    map<Label,IslandStats>::iterator it;
    for ( it = stats.subIslands.begin() ; it != stats.subIslands.end(); it++ ) {

        IslandStats & mainIsland = it->second;
        unsigned mainIslandSize = mainIsland.count;

        unsigned subIslandsToProcess = 0;

        map<Label,IslandStats>::iterator subIt;
        for (
            subIt = mainIsland.subIslands.begin();
            subIt != mainIsland.subIslands.end();
            subIt++
        ) {
            unsigned subIslandSize = subIt->second.count;

            float volumeRatio = subIslandSize / (float) mainIslandSize;

            if (volumeRatio > 0.001 && subIslandSize > 100) {
                subIt->second.active = true;
                subIslandsToProcess++;
            }
        }

        // if there are at least 2 sub-islands to separate,
        // process this main island
        if (subIslandsToProcess >= 2) {
            mainIsland.active = true;
        }
    }
}











bool LabelSizeSortPredicate(const PairLabelSize& a, const PairLabelSize& b) {
  return a.second < b.second;
}


/**
Return vector of island-labels sorted by the size in increasing order.
If countOnlyActive is true, then only
*/
vector<Label> getIslandLabelsSortedBySize(
    const map<Label, IslandStats> & islandMap,
    bool countOnlyActive
) {


    vector<PairLabelSize> labelSizeVector;

    map<Label,IslandStats>::const_iterator it;
    for (it = islandMap.begin(); it != islandMap.end(); it++) {
        Label label = it->first;
        unsigned size = it->second.count;
        bool active = it->second.active;

        if (countOnlyActive && !active)
            continue;

        labelSizeVector.push_back(PairLabelSize(label,size));
    }

    // sort the label-size vector according to size
    std::sort(
        labelSizeVector.begin(), labelSizeVector.end(), LabelSizeSortPredicate);

    // extract only labels from the vector
    vector<Label> sortedLabels;
    for (unsigned i=0; i<labelSizeVector.size(); ++i)
        sortedLabels.push_back(labelSizeVector[i].first);

    return sortedLabels;


}






/**
True if there's a pixel with intensity @label in the image
whose distance is less than @maxDistance.
*/
bool isIslandWithinDistance(
    UIntImagePtr image,
    FloatImagePtr distanceImage,
    Label label, float maxDistance
) {
    itk::ImageRegionIteratorWithIndex<UIntImage> it(
        image, image->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

        if (it.Get() != label)
            continue;

        float distance = distanceImage->GetPixel(it.GetIndex());

        if (distance < maxDistance)
            return true;

    }
    return false;
}










/**
Distinguish which subIslands are close to each other enough so it makes
sense to find bottleneck between them.

Two islands are considered "close to each other" if their
distance is smaller than maxDistance
*/
vector<pair<Label, Label> > getSubIslandsPairsForSeparation(
    const IslandStats &stats,
    UIntImagePtr subIslandsImage,
    unsigned maxDistance
) {

    vector<pair<Label, Label> > pairs;

    // each main label
    const map<Label, IslandStats> & mainIslandsMap = stats.subIslands;
    map<Label,IslandStats>::const_iterator it;
    for ( it = mainIslandsMap.begin() ; it != mainIslandsMap.end(); it++ ) {

        IslandStats  mainIslandStats = it->second;

        // skip islands which should not be process
        if (mainIslandStats.active == false)
            continue;

        // get sub-islands labels sorted by size from smallest to largest
        vector<Label> subIslandsSortedBySize =
          getIslandLabelsSortedBySize(mainIslandStats.subIslands, true);

        assert(subIslandsSortedBySize.size() >= 2);

        for (
            unsigned mainIdx=0;
            mainIdx<subIslandsSortedBySize.size() -1;
            ++mainIdx
        ) {

            Label subIsland = subIslandsSortedBySize[mainIdx];
            log("Computing distace from sub-island %d") % subIsland;

            FloatImagePtr distance =
                FilterUtils<UIntImage,FloatImage>::distanceMapByFastMarcher(
                    subIslandsImage, subIsland, maxDistance + 1);

            for (unsigned i=mainIdx+1; i < subIslandsSortedBySize.size(); ++i) {
                Label potentialAdjacentSubIsland = subIslandsSortedBySize[i];

                bool islandsAdjacent = isIslandWithinDistance(
                    subIslandsImage, distance,
                    potentialAdjacentSubIsland, maxDistance
                );

                if (islandsAdjacent) {
                    pairs.push_back(
                        pair<Label,Label>(subIsland,potentialAdjacentSubIsland)
                    );
                }
            }
        }
    }

    return pairs;
}






class DataCostFunction: public GCSegm::DataCostFunction {

private:
    Label subIslandLabels[2];
    UIntImagePtr subIslands;

public:

    // constructor
    DataCostFunction(UIntImagePtr islandImage, Label label1, Label label2) {
        subIslands = islandImage;
        subIslandLabels[0] = label1;
        subIslandLabels[1] = label2;
    }


    virtual int compute(ImageIndex idx, Label label) {

        Label labelInImage = subIslands->GetPixel(idx);
        Label labelInArgument = subIslandLabels[label];

        if (labelInImage == labelInArgument)
            return 1000;
        else
            return 0;
    }
};




class SmoothCostFunction : public GCSegm::SmoothnessCostFunction {
public:
    virtual int compute(ImageIndex idx1, ImageIndex idx2) {
        return 1;
    }
};







/**
Pixels which are labelled as 1 in segmentedBone image will
be assigned a new unique label in the resultImage
*/
void updateResult(UIntImagePtr resultImage, UIntImagePtr segmentedBone) {

    // find unique label as a maximum value of a label in the image plus one
    Label uniqueLabel = 0;
    itk::ImageRegionIterator<UIntImage> it(
        resultImage, resultImage->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        Label pixelLabel = it.Get();
        if (pixelLabel >= uniqueLabel)
            uniqueLabel = pixelLabel + 1;
    }

    // relabel result image
    itk::ImageRegionIteratorWithIndex<UIntImage> itBone(
        segmentedBone, segmentedBone->GetLargestPossibleRegion());
    for (itBone.GoToBegin(); !itBone.IsAtEnd(); ++itBone) {
        if (itBone.Get() == 1)
            resultImage->SetPixel(itBone.GetIndex(), uniqueLabel);
    }


}





/**
Find corresponding main island to a given subisland
*/
Label findLabelOfMainIsland(
    UIntImagePtr mainIslands,
    UIntImagePtr subIslands,
    Label subIslandLabel
) {

    itk::ImageRegionIteratorWithIndex<UIntImage> it(
        subIslands, subIslands->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

        if (it.Get() == subIslandLabel)
            return mainIslands->GetPixel(it.GetIndex());
    }

    assert(false);

}









/**
Input: Binary image
Output: Labelled image with detected bottlenecks between connected components

Nomenclature:
  mainIslands - connected components in the input binary image
  subIslands - connected components in the eroded input image
*/
UCharImagePtr compute(UCharImagePtr inputBinary) {

    unsigned EROSION_RADIUS = 3;
    unsigned MAX_DISTANCE_FOR_ADJACENT_BONES = 15;

    log("Computing Connected Components");
    UIntImagePtr mainIslands =
        FilterUtils<UCharImage,UIntImage>::connectedComponents(inputBinary);

    log("Erosion + Connected Components, ball radius=%d") % EROSION_RADIUS;
    UIntImagePtr subIslands =
        FilterUtils<UCharImage,UIntImage>::connectedComponents(
            FilterUtils<UCharImage>::erosion(inputBinary, EROSION_RADIUS)
        );


    /*
    In the following, we obtain pairs of sub-island labels between which the
    bottleneck should be found. E.g. [(5,2),(5,17),(13,12)] means
    we want to find the bottlenecks between sub-islands 5 and 2, between 5 and 17
    and between 13 and 12 -> three bottlenecks altogether. Clearly,
    the subislands 5 and 2 must lie within the same main island.
    */
    log("Discovering main islands containg bottlenecks");
    IslandStats stats = countSizeOfIslands(mainIslands, subIslands);
    markIslandsToProcess(stats);
    vector<pair<unsigned, unsigned> > subIslandsPairs =
        getSubIslandsPairsForSeparation(
            stats, subIslands, MAX_DISTANCE_FOR_ADJACENT_BONES);

    // prepare the result image
    UIntImagePtr result = ImageUtils<UIntImage>::duplicate(mainIslands);


    log("Number of bottlenecks to be found: %d") % subIslandsPairs.size();

    /*
    Let's find the bottlenecks using simplified graph-cut
    */
    for (unsigned i=0; i<subIslandsPairs.size(); ++i) {

        Label i1 = subIslandsPairs[i].first;
        Label i2 = subIslandsPairs[i].second;

        Label mainLabel = findLabelOfMainIsland(mainIslands, subIslands, i1);
        assert(mainLabel == findLabelOfMainIsland(mainIslands, subIslands, i2));

        log("Identifying bottleneck between sub-islands %d and %d within main island %d")
            % i1 % i2 % mainLabel;

        // for the graph-cut we need to supply roi and the cost function
        UIntImagePtr roi = FilterUtils<UIntImage>::binaryThresholding(
            mainIslands, mainLabel,mainLabel);
        DataCostFunction dataCostFunction(subIslands, i1, i2);
        SmoothCostFunction smoothCostFunction;

        // graph-cut segmentation
        GCSegm gcSegm;
        UIntImagePtr gcOutput =
            gcSegm.optimize(roi, &dataCostFunction, &smoothCostFunction);

        // update the result image
        updateResult(result, gcOutput);
    }


    // before we are finished, relabel componenets of the result according to
    // the size, i.e. 0 - background, 1 - largest, 2 - second largest, ...
    return FilterUtils<UIntImage,UCharImage>::relabelComponents(result);

}







}
