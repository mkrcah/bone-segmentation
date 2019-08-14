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

#define AVAILABLE_MEMORY_IN_MB  200


#include "itkImage.h"

#include "boost/format.hpp"
#include "boost/timer.hpp"

const unsigned int Dimension = 3;

typedef itk::Image< float, Dimension > FloatImage;
typedef itk::Image< short, Dimension > ShortImage;
typedef itk::Image< char, Dimension > CharImage;
typedef itk::Image< int, Dimension > IntImage;
typedef itk::Image< unsigned char, Dimension > UCharImage;
typedef itk::Image< unsigned int, Dimension > UIntImage;

typedef FloatImage::Pointer FloatImagePtr;
typedef ShortImage::Pointer ShortImagePtr;
typedef CharImage::Pointer  CharImagePtr;
typedef UCharImage::Pointer UCharImagePtr;
typedef UIntImage::Pointer  UIntImagePtr;
typedef IntImage::Pointer  IntImagePtr;

typedef UCharImage::SizeType ImageSize;
typedef UCharImage::IndexType ImageIndex;
typedef UCharImage::RegionType ImageRegion;

#define logger mylog << boost::format
#define logSetStage(stage) mylog.setStage(stage)

class MyLog {

   boost::timer m_timer;
   std::string m_stage;

public:

   template <class T> void operator<<(T t) {

       unsigned int elapsed = m_timer.elapsed();

       boost::format logLine("%2d:%02d [%15s] - %s\n");
       logLine % (elapsed / 60);
       logLine % (elapsed % 60);
       logLine % m_stage;
       logLine % t;

       std::clog << logLine;
   }

    void setStage(std::string stage) {
        m_stage = stage;
    }

};
MyLog mylog;
