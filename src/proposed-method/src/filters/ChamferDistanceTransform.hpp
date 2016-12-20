/*=========================================================================

    Program: Automatic Segmentation Of Bones In 3D-CT Images

    Author:  Marcel Krcah <marcel.krcah@gmail.com>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2010-09-01

    Version: 1.0

=========================================================================*/




#ifndef __chamfer_distance_h
#define __chamfer_distance_h

#include <vector>
#include "itkImageRegionIteratorWithIndex.h"
#include "ImageUtils.hpp"
#include "Globals.hpp"

/*

Implementation of Two-Sweep Chamfer Distance Transform as described in:

 [1] 3D Distance Fields: A Survey of Techniques and Applications,
     Mark W.Jones, J.Andreas Berentzen, Milos Sramek,
     IEEE Transactions on Visualization, 2006

*/
template<class LabelImage, class DistanceImage, class PropagationImage = DistanceImage>
class ChamferDistanceTransform {

public:

    enum DistanceType {
        MANHATTEN, // 3 positions
        CHESSBOARD, // 9 positions
        QUASI_EUCLIDEAN, // 9 positions
        COMPLETE_EUCLIDEAN // 13 positions
    };


private:

    typedef typename DistanceImage::Pointer DistanceImagePointer;
    typedef typename LabelImage::Pointer LabelImagePointer;
    typedef typename PropagationImage::Pointer PropagationImagePointer;

    typedef typename LabelImage::PixelType Label;
    typedef typename DistanceImage::PixelType Distance;
    typedef typename PropagationImage::PixelType PropagationPixel;

    typedef typename DistanceImage::IndexType ImageIndex;
    typedef typename DistanceImage::SizeType ImageRegionSize;
    typedef typename DistanceImage::RegionType ImageRegion;
    typedef typename DistanceImage::OffsetType ImageOffset;


    ImageRegion _largestRegion;
    ImageRegionSize _largestRegionSize;
    float _infinityDistance;
    PropagationImagePointer _propagationImage;

    struct TemplateElement {
        ImageOffset offset;
        float weight;

        TemplateElement(int x, int y, int z, float w) {
            weight = w;
            offset[0] = x;
            offset[1] = y;
            offset[2] = z;
        }
    };

    typedef std::vector<TemplateElement> ChamferTemplate;


    void addToTemplateIfPositiveWeight(
        ChamferTemplate & templ, int x, int y, int z, float weight) {

        // do nothing for zero weight
        if (weight < 0000.1 )
            return;

        templ.push_back(TemplateElement(x,y,z, weight));
    }

    /*
    Pre-compude chamfer forward distance template
    */
    ChamferTemplate getForwardTemplate(DistanceType type) {

        ChamferTemplate templ;

        float a,b,c;

        // set weight according to the distance type
        switch (type) {
            case MANHATTEN:
                a = 1.0; b = 0.0; c = 0.0;
                break;
            case CHESSBOARD:
                a = 1.0; b = 1.0; c = 0.0;
                break;
            case QUASI_EUCLIDEAN:
                a = 1.0; b = sqrt(2); c = 0.0;
                break;
            case COMPLETE_EUCLIDEAN:
                a = 1.0; b = sqrt(2); c = sqrt(3);
                break;
            default:
                assert(false);
        }


        addToTemplateIfPositiveWeight(templ, -1,  0,  0, a);
        addToTemplateIfPositiveWeight(templ,  0, -1,  0, a);
        addToTemplateIfPositiveWeight(templ, -1, -1,  0, b);
        addToTemplateIfPositiveWeight(templ, -1, +1,  0, b);

            addToTemplateIfPositiveWeight(templ,  0,  0, -1, a);

            addToTemplateIfPositiveWeight(templ, -1,  0, -1, b);
            addToTemplateIfPositiveWeight(templ, +1,  0, -1, b);
            addToTemplateIfPositiveWeight(templ,  0, -1, -1, b);
            addToTemplateIfPositiveWeight(templ,  0, +1, -1, b);

            addToTemplateIfPositiveWeight(templ, -1, -1, -1, c);
            addToTemplateIfPositiveWeight(templ, -1, +1, -1, c);
            addToTemplateIfPositiveWeight(templ, +1, -1, -1, c);
            addToTemplateIfPositiveWeight(templ, +1, +1, -1, c);

        return templ;

    }


    /*
                           /  infty       for labelImg[idx] = 0,
    InitialDistance[idx] =
                           \  0           otherwise,

    where idx denotes index of a pixel.
    */
    DistanceImagePointer initializeDistanceTransform(LabelImagePointer labelImg) {

        DistanceImagePointer distanceImg =
            FilterUtils<LabelImage,DistanceImage>::createEmptyFrom(labelImg);

        itk::ImageRegionIteratorWithIndex<LabelImage> it(labelImg, _largestRegion);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {

            distanceImg->SetPixel(
                it.GetIndex(),
                (it.Get() == 0) ? _infinityDistance : 0
            );
        }

        return distanceImg;
    }






    void updatePixel(
        DistanceImagePointer distMap,
        ImageIndex centerPixelIndex,
        const ChamferTemplate &templ
    ) {

        float minDistance = distMap->GetPixel(centerPixelIndex);
        ImageIndex minIndex = centerPixelIndex;

        for (unsigned i = 0; i < templ.size(); ++i) {

            const TemplateElement &elem = templ[i];

            ImageIndex idx = centerPixelIndex + elem.offset;
            if (_largestRegion.IsInside(idx)) {

                float d = elem.weight + distMap->GetPixel(idx);
                if (d < minDistance) {
                    minDistance = d;
                    minIndex = idx;
                }
            }
        }

        if (!_propagationImage.IsNull()) {
            _propagationImage->SetPixel(
                centerPixelIndex,
                _propagationImage->GetPixel(minIndex)
            );
        }
        distMap->SetPixel(centerPixelIndex, minDistance);
    }





    void forwardSweep(DistanceImagePointer distMap,ChamferTemplate templ) {

        // perform forward sweep
        itk::ImageRegionIteratorWithIndex<DistanceImage> it(distMap, _largestRegion);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            updatePixel(distMap, it.GetIndex(), templ);
        }
    }




    void backwardSweep(DistanceImagePointer distMap,ChamferTemplate templ) {

        // reverse chamfer template
        for (unsigned elemIndex = 0; elemIndex < templ.size(); ++elemIndex) {
           for (unsigned dim=0; dim < ImageOffset::GetOffsetDimension(); ++ dim) {
               templ[elemIndex].offset[dim] *= -1;
           }
        }

        // perform backward sweep
        itk::ImageRegionIteratorWithIndex<DistanceImage> it(distMap, _largestRegion);
        for (it.GoToReverseBegin(); !it.IsAtReverseEnd(); --it) {
            updatePixel(distMap,it.GetIndex(), templ);
        }

    }





public:

    // constructor
    ChamferDistanceTransform() : _propagationImage(NULL)
    {}

    void setPropagationImage(PropagationImagePointer p) {
        _propagationImage = p;
    }

    PropagationImagePointer getPropagationImage() {
        return _propagationImage;
    }



    string getDistanceTypeDescr(DistanceType type) {

        switch (type) {
            case MANHATTEN:
                return "Manhattan";
            case CHESSBOARD:
                return "Chessboard";
            case QUASI_EUCLIDEAN:
                return "QuasiEuclidean";
            case COMPLETE_EUCLIDEAN:
                return "CompleteEuclidean";
            default:
                assert(false);
        }
    }






    /*
    Input binary image,
    Output chamfer distance as a floating point image
    */
    DistanceImagePointer compute(LabelImagePointer labelImg, DistanceType type) {


        // initialize variables used all over the computation
        _largestRegion = labelImg->GetLargestPossibleRegion();
        _largestRegionSize = _largestRegion.GetSize();
        _infinityDistance = _largestRegionSize[0] + _largestRegionSize[1]
            + _largestRegionSize[2] + 1;

        DistanceImagePointer distanceMap = initializeDistanceTransform(labelImg);

        // compute the template for the given distance transfortm type
        ChamferTemplate chamferTemplate = getForwardTemplate(type);

        log("Chamfer Distance Forward sweep");
        forwardSweep(distanceMap, chamferTemplate);

        log("Chamfer Distance Backward sweep");
        backwardSweep(distanceMap, chamferTemplate);

        // done :)
        return distanceMap;
    }


};

#endif
