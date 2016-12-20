/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/






#include "Globals.hpp"
#include <vector>
#include "boost/tuple/tuple.hpp"

const unsigned MAX_BLOCKS = 100;
const unsigned OVERLAP = 5;


using std::vector;



template<class Image>
class ImageSplitter {



private:

    typedef typename Image::Pointer ImagePointer;


    struct SliceSet {

        // index of the first slice in the subset
        unsigned int begin;

        // index of the last slice in the subset
        unsigned int end;

    };




    static vector<SliceSet> partitionSlices(
        unsigned blocks, unsigned slicesTotal, unsigned overlap
    ) {

        vector<SliceSet> sliceSets;

        unsigned baseSize =  slicesTotal / blocks;
        for (unsigned i=0; i<blocks; ++i) {
            SliceSet s;
            s.begin = std::max(0,(int)(i * baseSize - overlap));
            s.end = std::min(slicesTotal-1, (i+1) * baseSize + overlap);
            sliceSets.push_back(s);
        }
        return sliceSets;
    }





    static unsigned getDirectionWithMaxSize(ImagePointer image) {

        ImageSize size = image->GetLargestPossibleRegion().GetSize();

        unsigned maxSizeIndex = 0;
        for (unsigned i=0; i<image->GetImageDimension(); ++i) {
            if (size[i] > size[maxSizeIndex]) {
                maxSizeIndex = i;
            }
        }
        return maxSizeIndex;
    }





    /**
    Calculate number of object pixels in the image in each slice
    along the given direction.
    */
    static vector<unsigned> getNumberOfSeedsForEachSlice(
        ImagePointer image, unsigned axis
    ) {

        unsigned totalSlices = image->GetLargestPossibleRegion().GetSize()[axis];

        vector<unsigned> seedsInSlices(totalSlices, 0);

        itk::ImageRegionIteratorWithIndex<Image> it(
            image, image->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            assert(it.Get() == 0 || it.Get() == 1);

            if (it.Get() == 1) {
                seedsInSlices[it.GetIndex()[axis]]++;
            }

        }

        return seedsInSlices;
    }



    /** Sum elements of a vector from the index @from to index @to */
    static unsigned sumVectorElements(
        const vector<unsigned> &v, unsigned from, unsigned to
    ) {
        unsigned sum = 0;
        for (unsigned idx = from; idx <= to; ++idx) {
            sum += v[idx];
        }
        return sum;
    }




    static  vector<SliceSet> splitAlongAxis(
        ImagePointer labelImage, unsigned axis
    ) {

        unsigned slices = labelImage->GetLargestPossibleRegion().GetSize()[axis];

        // calculate number of non-zero pixels in each slice
        vector<unsigned> seedsInSlices =
            getNumberOfSeedsForEachSlice(labelImage, axis);
        assert(seedsInSlices.size() == slices);

        /*
        In the rest of the function we find the minimum number of blocks
        such that for each block the graph for the graph-cut segmentation
        may be computed using max @AVAILABLE_MEMORY_IN_MB memory RAM.
        */

        bool enoughRAMforEachBlock = false;
        unsigned blocks = 0;

        unsigned availableMemoryKb = AVAILABLE_MEMORY_IN_MB * 1024;

        unsigned memoryNeededKb;
        vector<SliceSet> sliceSets;

        log("Maximum available memory is %d MB") % AVAILABLE_MEMORY_IN_MB;


        while (!enoughRAMforEachBlock && blocks < MAX_BLOCKS) {
            ++blocks;

            sliceSets = partitionSlices(blocks, slices, 0);

            unsigned maxPixelsInOneBlock = 0;

            for (unsigned iBlock=0; iBlock < blocks; ++iBlock) {
                unsigned seedsInBlock = sumVectorElements(
                    seedsInSlices,
                    sliceSets[iBlock].begin,
                    sliceSets[iBlock].end);
                maxPixelsInOneBlock = std::max(maxPixelsInOneBlock, seedsInBlock);
            }


            bool is32bit = (sizeof(void*) == 4);
            memoryNeededKb = maxPixelsInOneBlock * ( is32bit ? 124 : 232 );
            memoryNeededKb/=1024;

            enoughRAMforEachBlock = (memoryNeededKb < availableMemoryKb);

            log("Probing %1% block(s): %2%, graph-cut expected memory consumption %3% Mb")
                % blocks
                % (enoughRAMforEachBlock ? "OK" : "Not enough")
                % (memoryNeededKb / 1024);


        } // while

        return sliceSets;
    }



    static ImageRegion sliceSetToRegion(
        SliceSet sliceSet, unsigned axis, ImageSize imageSize
    ) {

        ImageRegion region;

        // region size
        ImageSize regionSize = imageSize;
        regionSize[axis] = sliceSet.end - sliceSet.begin + 1;
        region.SetSize(regionSize);

        // starting index of the region
        ImageIndex startIndex;
        startIndex.Fill(0);
        startIndex[axis] = sliceSet.begin;
        region.SetIndex(startIndex);

        return region;

    }





public:

    static vector<ImageRegion>  splitIntoRegions(ImagePointer image) {

        unsigned axis = getDirectionWithMaxSize(image);

        vector<SliceSet> sliceSets = splitAlongAxis(image,axis);

        // prepare result
        vector<ImageRegion> regions;

        ImageSize imageSize = image->GetLargestPossibleRegion().GetSize();
        for (unsigned i = 0; i < sliceSets.size(); ++i) {
            regions.push_back(sliceSetToRegion(sliceSets[i], axis, imageSize));
        }

        return regions;
    }


};














