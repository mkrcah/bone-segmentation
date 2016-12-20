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


#include "itkImage.h"
#include "itkImageDuplicator.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMaskNegatedImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkShiftScaleImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkPasteImageFilter.h"

#include "ImageUtils.hpp"
#include <algorithm> //max,min

template<class InputImage, class OutputImage = InputImage>
class FilterUtils {

    typedef typename InputImage::Pointer  InputImagePointer;
    typedef typename InputImage::PixelType InputImagePixelType;
    typedef typename InputImage::IndexType InputImageIndex;
    typedef typename InputImage::RegionType InputImageRegion;

    typedef typename OutputImage::Pointer OutputImagePointer;
    typedef typename OutputImage::IndexType OutputImageIndex;
    typedef typename OutputImage::RegionType OutputImageRegion;
    typedef typename OutputImage::PixelType OutputImagePixelType;


    typedef itk::FlatStructuringElement< InputImage::ImageDimension > StructuringElementType;
    typedef typename StructuringElementType::RadiusType StructuringElementTypeRadius;

    typedef itk::BinaryThresholdImageFilter<InputImage,OutputImage> BinaryThresholdFilter;
    typedef itk::BinaryErodeImageFilter<InputImage, OutputImage, StructuringElementType > ErodeFilterType;
    typedef itk::BinaryDilateImageFilter<InputImage, OutputImage, StructuringElementType > DilateFilterType;
    typedef itk::CastImageFilter <InputImage,OutputImage> CastImageFilterType;
    typedef itk::MaskImageFilter<InputImage,InputImage,OutputImage> MaskImageFilterType;
    typedef itk::MaskNegatedImageFilter<InputImage,InputImage,OutputImage> MaskNegatedImageFilterType;
    typedef itk::SubtractImageFilter <InputImage,InputImage,OutputImage> SubtractFilterType;
    typedef itk::AddImageFilter <InputImage,InputImage,OutputImage> AddFilterType;
    typedef itk::ConnectedComponentImageFilter<InputImage,OutputImage>  ConnectedComponentImageFilterType;
    typedef itk::RelabelComponentImageFilter<InputImage,OutputImage>  RelabelComponentImageFilterType;
    typedef itk::ShiftScaleImageFilter<InputImage,OutputImage>  ShiftScaleImageFilterType;
    typedef itk::DiscreteGaussianImageFilter<InputImage,OutputImage>  DiscreteGaussianImageFilterType;
    typedef itk::FastMarchingImageFilter<OutputImage>  FastMarchingImageFilterType;
    typedef itk::PasteImageFilter<InputImage,OutputImage>  PasteImageFilterType;

    typedef typename BinaryThresholdFilter::Pointer BinaryThresholdFilterPointer;
    typedef typename ErodeFilterType::Pointer ErodeFilterPointer;
    typedef typename DilateFilterType::Pointer DilateFilterPointer;
    typedef typename CastImageFilterType::Pointer CastFilterPointer;
    typedef typename MaskImageFilterType::Pointer MaskImageFilterPointer;
    typedef typename MaskNegatedImageFilterType::Pointer MaskNegatedImageFilterPointer;
    typedef typename SubtractFilterType::Pointer SubtractFilterPointer;
    typedef typename AddFilterType::Pointer AddFilterPointer;
    typedef typename ConnectedComponentImageFilterType::Pointer ConnectedComponentImageFilterPointer;
    typedef typename RelabelComponentImageFilterType::Pointer RelabelComponentImageFilterPointer;
    typedef typename ShiftScaleImageFilterType::Pointer ShiftScaleImageFilterPointer;
    typedef typename DiscreteGaussianImageFilterType::Pointer DiscreteGaussianImageFilterPointer;
    typedef typename FastMarchingImageFilterType::Pointer FastMarchingImageFilterPointer;
    typedef typename PasteImageFilterType::Pointer PasteImageFilterPointer;

    typedef ImageUtils<InputImage> InputImageUtils;
    typedef ImageUtils<OutputImage> OutputImageUtils;

    typedef typename FastMarchingImageFilterType::NodeContainer FastMarchingNodeContainer;
    typedef typename FastMarchingNodeContainer::Pointer FastMarchingNodeContainerPointer;
    typedef typename FastMarchingImageFilterType::NodeType FastMarchingNodeType;






public:




    /**
    Paste the region from the source image to a given position
    in the destination image
    */
    static OutputImagePointer paste(
        InputImagePointer sourceImage, InputImageRegion sourceRegion,
        OutputImagePointer destinationImage, OutputImageIndex destinationIndex
    ) {

        PasteImageFilterPointer filter =
            PasteImageFilterType::New();

        filter->SetSourceImage(sourceImage);
        filter->SetSourceRegion(sourceRegion);
        filter->SetDestinationImage(destinationImage);
        filter->SetDestinationIndex(destinationIndex);

        filter->Update();

        return filter->GetOutput();
    }







    // output_pixel =  scale * input_pixel + shift
    static OutputImagePointer linearTransform(
        InputImagePointer image,
        OutputImagePixelType scale,
        OutputImagePixelType shift = 0
    ) {

        ShiftScaleImageFilterPointer filter =
            ShiftScaleImageFilterType::New();

        filter->SetInput(image);
        filter->SetScale(scale);
        filter->SetShift(shift);
        filter->Update();
        return filter->GetOutput();
    }


    // smooth the image with a discrete gaussian filter
    static OutputImagePointer gaussian(
        InputImagePointer image, float variance
    ) {

        DiscreteGaussianImageFilterPointer filter =
            DiscreteGaussianImageFilterType::New();

        filter->SetInput(image);
        filter->SetVariance(variance);
        filter->Update();

        return filter->GetOutput();
    }




    // relabel components according to its size.
    // Largest component 1, second largest 2, ...
    static OutputImagePointer relabelComponents(InputImagePointer image) {

        RelabelComponentImageFilterPointer filter =
            RelabelComponentImageFilterType::New();

        filter->SetInput(image);
        filter->Update();

        return filter->GetOutput();
    }



    // cast the image to the output type
    static OutputImagePointer cast(InputImagePointer image) {
        CastFilterPointer castFilter = CastImageFilterType::New();
        castFilter->SetInput(image);
        castFilter->Update();
        return castFilter->GetOutput();
    }





    static OutputImagePointer createEmptyFrom(
        InputImagePointer input
    ) {

        OutputImagePointer output = OutputImageUtils::createEmpty(
            input->GetLargestPossibleRegion().GetSize());
        output->SetOrigin(input->GetOrigin());
        output->SetSpacing(input->GetSpacing());
        output->SetDirection(input->GetDirection());
        output->FillBuffer(0);

        return output;

    }



    static OutputImagePointer substract(
        InputImagePointer image1, InputImagePointer image2
    ) {
        SubtractFilterPointer substractFilter = SubtractFilterType::New();
        substractFilter->SetInput1(image1);
        substractFilter->SetInput2(image2);
        substractFilter->Update();
        return substractFilter->GetOutput();
    }





    // pixel-wise addition of two images
    static OutputImagePointer add(
        InputImagePointer image1, InputImagePointer image2
    ) {
        AddFilterPointer addFilter = AddFilterType::New();
        addFilter->SetInput1(image1);
        addFilter->SetInput2(image2);
        addFilter->Update();
        return addFilter->GetOutput();
    }




    static OutputImagePointer mask(
        InputImagePointer image,
        InputImagePointer mask
    ) {
        MaskImageFilterPointer filter = MaskImageFilterType::New();

        filter->SetInput1(image);
        filter->SetInput2(mask);
        filter->Update();

        return filter->GetOutput();
    }





    static OutputImagePointer negatedMask(
        InputImagePointer image,
        InputImagePointer mask
    ) {
        MaskNegatedImageFilterPointer filter = MaskNegatedImageFilterType::New();

        filter->SetInput1(image);
        filter->SetInput2(mask);
        filter->Update();

        return filter->GetOutput();
    }





    static OutputImagePointer binaryThresholding(
        InputImagePointer inputImage,
        InputImagePixelType lowerThreshold,
        InputImagePixelType upperThreshold,
        OutputImagePixelType insideValue = 1,
        OutputImagePixelType outsideValue = 0
    ) {
        BinaryThresholdFilterPointer thresholder = BinaryThresholdFilter::New();
        thresholder->SetInput(inputImage);

        thresholder->SetLowerThreshold( lowerThreshold );
        thresholder->SetUpperThreshold( upperThreshold );
        thresholder->SetInsideValue(insideValue);
        thresholder->SetOutsideValue(outsideValue);

        thresholder->Update();

        return thresholder->GetOutput();
    }





    // assign pixels with intensities above upperThreshold
    // value upperThreshold and below lowerThreshold value
    // lowerThreshold
    static OutputImagePointer thresholding(
        InputImagePointer inputImage,
        InputImagePixelType lowerThreshold,
        InputImagePixelType upperThreshold
    ) {

        itk::ImageRegionIterator<InputImage> it(
            inputImage, inputImage->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            it.Set(
                std::min(
                    upperThreshold,
                    std::max(it.Get(),lowerThreshold)
                )
            );
        }

        return cast(inputImage);
    }




    // perform erosion (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer erosion(
        InputImagePointer labelImage, unsigned radius,
        InputImagePixelType valueToErode = 1
    ) {

        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );

        ErodeFilterPointer erosionFilter = ErodeFilterType::New();
        erosionFilter->SetKernel(K);
        erosionFilter->SetErodeValue(valueToErode);

        erosionFilter->SetInput( labelImage );
        erosionFilter->Update();

        return erosionFilter->GetOutput();
    }




    // perform erosion (mathematical morphology) with a given label image
    // using a ball with a given radius
    static OutputImagePointer dilation(
        InputImagePointer labelImage, unsigned radius,
        InputImagePixelType valueToDilate = 1
    ) {
        StructuringElementTypeRadius rad; rad.Fill(radius);
        StructuringElementType K = StructuringElementType::Ball( rad );

        DilateFilterPointer dilateFilter  = DilateFilterType::New();
        dilateFilter->SetKernel(K);
        dilateFilter->SetDilateValue(valueToDilate);

        dilateFilter->SetInput( labelImage );
        dilateFilter->Update();

        return dilateFilter->GetOutput();
    }



    // compute connected components of a (binary image)
    static OutputImagePointer connectedComponents(InputImagePointer image) {

        ConnectedComponentImageFilterPointer filter =
            ConnectedComponentImageFilterType::New();

        filter->SetInput(image);
        filter->Update();

        return filter->GetOutput();
    }




    /**
    Compute distance from the object using fast marching front.

    All pixels with value objectLabel are considered to belong to an object.
    If positive stopping value is specified, the maximum computed distance
    will be stoppingValue, otherwise the distance will be computed for the
    whole image. The image spacing is ignored, 1 is used for all directions.
    */
    static OutputImagePointer distanceMapByFastMarcher(
        InputImagePointer image,
        InputImagePixelType objectLabel,
        float stoppingValue = 0
    ) {

        // prepare fast marching
        FastMarchingImageFilterPointer fastMarcher =
            FastMarchingImageFilterType::New();
        fastMarcher->SetOutputSize(image->GetLargestPossibleRegion().GetSize());
        fastMarcher->SetOutputOrigin(image->GetOrigin() );
        fastMarcher->SetOutputSpacing(image->GetSpacing() );
        fastMarcher->SetOutputDirection(image->GetDirection() );
        fastMarcher->SetSpeedConstant(1.0);
        if (stoppingValue > 0)
            fastMarcher->SetStoppingValue(stoppingValue);

        // set seeds as pixels in the island @label
        FastMarchingNodeContainerPointer seeds = FastMarchingNodeContainer::New();
        seeds->Initialize();

        itk::ImageRegionIteratorWithIndex<InputImage> it(
            image, image->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            if (it.Get() == objectLabel) {
                FastMarchingNodeType & node = seeds->CreateElementAt(seeds->Size());
                node.SetValue(0);
                node.SetIndex(it.GetIndex());
            }
        }
        fastMarcher->SetTrialPoints(seeds);

        // perform fast marching
        fastMarcher->Update();

        // done :)
        return  fastMarcher->GetOutput();

    }









};
