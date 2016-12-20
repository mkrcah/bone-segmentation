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
#include "ImageUtils.hpp"
#include "FilterUtils.hpp"
#include "SheetnessMeasure.hpp"
#include "ChamferDistanceTransform.hpp"
#include "boost/tuple/tuple.hpp"


namespace Preprocessing {

using namespace boost;


// compute multiscale sheetness measure
// if roi is specified, than compute the measure only for pixels within ROI
// (i.e. pixels where roi(pixel) >= 1)
FloatImagePtr multiscaleSheetness(
    FloatImagePtr img, vector<float> scales, UCharImagePtr roi = 0
) {

    assert(scales.size() >= 1);

    FloatImagePtr multiscaleSheetness;

    for (unsigned i = 0; i < scales.size(); ++i) {

        log("Computing single-scale sheetness, sigma=%4.2f") % scales[i];

        MemoryEfficientObjectnessFilter *sheetnessFilter =
            new MemoryEfficientObjectnessFilter();
        sheetnessFilter->SetImage(img);
        sheetnessFilter->SetAlpha(0.5);
        sheetnessFilter->SetBeta(0.5);
        sheetnessFilter->SetSigma(scales[i]);
        sheetnessFilter->SetObjectDimension(2);
        sheetnessFilter->SetBrightObject(true);
        sheetnessFilter->ScaleObjectnessMeasureOff();
        sheetnessFilter->Update();
        sheetnessFilter->SetROIImage(roi);

        FloatImagePtr singleScaleSheetness = sheetnessFilter->GetOutput();

        if (i==0) {
            multiscaleSheetness = singleScaleSheetness;
            continue;
        }

        // update the multiscale sheetness
        // take the value which is larger in absolute value
        itk::ImageRegionIterator<FloatImage>
            itMulti(singleScaleSheetness,singleScaleSheetness->GetLargestPossibleRegion());
        itk::ImageRegionIterator<FloatImage>
            itSingle(multiscaleSheetness,multiscaleSheetness->GetLargestPossibleRegion());
        for (
                itMulti.GoToBegin(),itSingle.GoToBegin();
                !itMulti.IsAtEnd();
                ++itMulti, ++itSingle
            ) {
                float multiVal = itMulti.Get();
                float singleVal = itSingle.Get();

                // higher absolute value is better
                if (abs(singleVal) > abs(multiVal)) {
                    itMulti.Set(singleVal);
                }
            }
    } // iteration trough scales

    return multiscaleSheetness;

}



FloatImagePtr chamferDistance(UCharImagePtr image) {
    typedef ChamferDistanceTransform<UCharImage, FloatImage> CDT;
    CDT cdt;
    return cdt.compute(image, CDT::MANHATTEN);
}



/*
Input: Normalized CT image, scales for the sheetness measure
Output: (ROI, MultiScaleSheetness, SoftTissueEstimation)
*/
tuple<UCharImagePtr, FloatImagePtr, UCharImagePtr>
compute(
    ShortImagePtr inputCT,
    float sigmaSmallScale,
    vector<float> sigmasLargeScale
) {

    UCharImagePtr roi;
    UCharImagePtr softTissueEstimation;
    FloatImagePtr sheetness;

    {
        log("Thresholding input image");
        ShortImagePtr thresholdedInputCT =
            FilterUtils<ShortImage>::thresholding(
                ImageUtils<ShortImage>::duplicate(inputCT),
                25, 600
            );

        vector<float> scales; scales.push_back(sigmaSmallScale);
        FloatImagePtr smallScaleSheetnessImage =
            multiscaleSheetness(
                FilterUtils<ShortImage,FloatImage>::cast(thresholdedInputCT),
                scales
            );

        log("Estimating soft-tissue voxels");
        softTissueEstimation =  FilterUtils<UIntImage,UCharImage>::binaryThresholding(
                FilterUtils<UIntImage>::relabelComponents(
                    FilterUtils<UCharImage, UIntImage>::connectedComponents(
                    FilterUtils<FloatImage,UCharImage>::binaryThresholding(
                        smallScaleSheetnessImage, -0.05, +0.05)
                    )),
                1,1
            );

        log("Estimating bone voxels");
        UCharImagePtr boneEstimation =
            FilterUtils<ShortImage,UCharImage>::createEmptyFrom(inputCT);
        itk::ImageRegionIteratorWithIndex<ShortImage>
            it(inputCT,inputCT->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            short hu = it.Get();
            float sheetness = smallScaleSheetnessImage->GetPixel(it.GetIndex());

            bool bone = (hu > 400) || ( hu > 250 && sheetness > 0.6 );

            boneEstimation->SetPixel(it.GetIndex(), bone ? 1 : 0);
        }

        log("Computing ROI from bone estimation using Chamfer Distance");
        roi = FilterUtils<FloatImage,UCharImage>::binaryThresholding(
            chamferDistance(boneEstimation),0, 30);
    }

    log("Unsharp masking");
    FloatImagePtr inputCTUnsharpMasked =
        FilterUtils<FloatImage>::add(
            FilterUtils<ShortImage,FloatImage>::cast(inputCT),
            FilterUtils<FloatImage>::linearTransform(
                FilterUtils<FloatImage>::substract(
                    FilterUtils<ShortImage,FloatImage>::cast(inputCT),
                    FilterUtils<ShortImage,FloatImage>::gaussian(inputCT, 1.0)),
                10.0, 0.0)
        );

    log("Computing multiscale sheetness measure at %d scales")
        % sigmasLargeScale.size();
    sheetness = multiscaleSheetness(inputCTUnsharpMasked, sigmasLargeScale, roi);


    return make_tuple(roi,sheetness, softTissueEstimation);
}





} // namespace Preprocessing
