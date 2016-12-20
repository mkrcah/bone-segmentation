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
#include "Globals.hpp"


namespace Segmentation {



typedef GraphCutSegmentation<Dimension> GCSegm;
typedef GCSegm::LabelID Label;


const Label BONE   = 1;
const Label TISSUE = 0;
const int COST_AMPLIFIER = 1000;




//==============================================================================
//    Cost Functions
//==============================================================================


class SheetnessBasedDataCost : public GCSegm::DataCostFunction {
private:

    ShortImagePtr intensity;
    FloatImagePtr sheetness;
    UCharImagePtr softTissueEstimation;

public:

    // constructor
    SheetnessBasedDataCost(
        ShortImagePtr p_intensity,
        FloatImagePtr p_sheetnessMeasure,
        UCharImagePtr p_softTissueEstimation
    )
    : intensity(p_intensity)
    , sheetness(p_sheetnessMeasure)
    , softTissueEstimation(p_softTissueEstimation)
    { /* empty body */}

    virtual int compute(ImageIndex idx, Label label) {


        short hu = intensity->GetPixel(idx);
        float s = sheetness->GetPixel(idx);
        unsigned char t = softTissueEstimation->GetPixel(idx);

        assert( t==0 || t==1);
        assert( s > -1.001 && s < 1.001);

        float totalCost;
        switch (label) {
            case BONE  :
                totalCost = (hu < -500 || t == 1) ? 1 : 0;
                break;

            case TISSUE:
                  totalCost = ( hu > 400) && ( s > 0 ) ? 1 : 0;
                break;

            default:
                assert(false);
        }

        assert(totalCost >= 0 || totalCost < 1.001);

        return COST_AMPLIFIER * totalCost;


    }

};



class SheetnessBasedSmoothCost : public GCSegm::SmoothnessCostFunction {
private:

    ShortImagePtr intensity;
    FloatImagePtr sheetness;

public:

    SheetnessBasedSmoothCost(
        ShortImagePtr p_intensity,
        FloatImagePtr p_sheetnessMeasure
    )
    : intensity(p_intensity)
    , sheetness(p_sheetnessMeasure)
    { /* empty body */}


    virtual int compute(
        ImageIndex idx1, ImageIndex idx2
    ) {

//        float hu1 = intensity->GetPixel(idx1);
//        float hu2 = intensity->GetPixel(idx2);
//        float dHU   = abs(hu1 - hu2);

        float s1 = sheetness->GetPixel(idx1);
        float s2 = sheetness->GetPixel(idx2);
        float dSheet = abs(s1 - s2);

        float sheetnessCost = (s1 < s2) ? 1.0 : exp ( - 5 * dSheet);

        float alpha = 5.0;
        return COST_AMPLIFIER * alpha *  sheetnessCost  + 1;

    }
};




//==============================================================================
//    Segmentation
//==============================================================================




UCharImagePtr compute(
    ShortImagePtr intensity,
    UCharImagePtr roi,
    FloatImagePtr sheetnessMeasure,
    UCharImagePtr softTissueEstimation
) {


    // data cost functions
    SheetnessBasedDataCost dataCostFunction(
        intensity, sheetnessMeasure, softTissueEstimation);
    SheetnessBasedSmoothCost smoothCostFunction(
        intensity,sheetnessMeasure);

    // graph-cut segmentation
    GCSegm gcSegm;
    UIntImagePtr gcOutput = gcSegm.optimize(
        FilterUtils<UCharImage,UIntImage>::cast(roi),
        &dataCostFunction, &smoothCostFunction
    );

    // finitto :)
    return FilterUtils<UIntImage,UCharImage>::cast(gcOutput);

}






}
