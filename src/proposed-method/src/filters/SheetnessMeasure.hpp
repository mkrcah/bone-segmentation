/*=========================================================================

    Author:  Rémi Blanc <blanc@vision.ee.ethz.ch>
             Computer Vision Laboratory
             ETH Zurich
             Switzerland

    Date:    2009-08-26
    Version: 1.0

=========================================================================*/


#pragma once


#include "itkImage.h"
#include "itkOrientedImage.h"

#include "itkImageRegionIterator.h"
#include "itkImportImageFilter.h"
#include "vnl/vnl_math.h"
#include <itkSmoothingRecursiveGaussianImageFilter.h>
#include "vnl/algo/vnl_symmetric_eigensystem.h"

#include <limits>
#include "Globals.hpp"


//#include "image_utils.h"


/*! \brief
*	Defines a structure allowing to define an shape from the corresponding shape model
*	shape parameters <=> 3D shape <=> vtk mesh
*/

class MemoryEfficientObjectnessFilter
{
public:
	typedef float PixelType;
	typedef itk::Image< PixelType, 3 > ImageType;
	typedef ImageType::Pointer	ImagePointerType;
    typedef itk::Vector<float,3> VectorType;
	typedef itk::Image< VectorType , 3 > VectorImageType;
	typedef VectorImageType::Pointer	VectorImagePointerType;
	typedef itk::Image<unsigned char,3> ROIImageType;
	typedef ROIImageType::Pointer	ROIImagePointerType;

	MemoryEfficientObjectnessFilter();

	void SetImage(ImagePointerType image);
	void SetVectorImage(VectorImagePointerType vectorImage);
	void SetROIImage(ROIImagePointerType roiImage);
	void SetObjectDimension(unsigned int dim);
	void SetAlpha(double a);
	void SetBeta(double b);
	void SetGamma(double c);
	void SetSigma(double s);
	void SetBrightObject(bool cond);

	void ScaleObjectnessMeasureOff();
	void ScaleObjectnessMeasureOn();

	void Update();
	ImagePointerType GetOutput();


private:
	ImagePointerType input_image, output_image;
	VectorImagePointerType vector_image;
	ROIImagePointerType roi_image;
	unsigned int objectDimension;
	double alpha, beta, gamma, sigma;
	float bright;
	bool scaleObjectnessMeasure;

	void Eigenvalues_3_3_symetric(
        float M11, float M12, float M13, float M22, float M23, float M33,
        VectorType & eigenVals);

    void solve_3x3_symmetric_eigensystem(
        float M11, float M12, float M13, float M22, float M23, float M33,
        VectorType & eigenVals,
        VectorType & firstPrincipalEigenvector);

	void GenerateObjectnessImage();
};



using namespace std;

MemoryEfficientObjectnessFilter::MemoryEfficientObjectnessFilter()
{
	alpha = 0.5;
	beta = 0.5;
	gamma = 0.5;
	sigma = 1;
	objectDimension = 2;
	scaleObjectnessMeasure = false;
	bright = 1;
}

//
//
//

void MemoryEfficientObjectnessFilter::SetImage(ImagePointerType image)		{ input_image = image; }
void MemoryEfficientObjectnessFilter::SetVectorImage(VectorImagePointerType image)		{ vector_image = image; }
void MemoryEfficientObjectnessFilter::SetROIImage(ROIImagePointerType image)		{ roi_image = image; }
void MemoryEfficientObjectnessFilter::SetObjectDimension(unsigned int dim)	{ objectDimension = dim; }
void MemoryEfficientObjectnessFilter::SetAlpha(double a)					{ alpha = a; }
void MemoryEfficientObjectnessFilter::SetBeta(double b)						{ beta = b; }
void MemoryEfficientObjectnessFilter::SetGamma(double c)					{ gamma = c; }
void MemoryEfficientObjectnessFilter::SetSigma(double s)					{ sigma = s; }
void MemoryEfficientObjectnessFilter::SetBrightObject(bool cond)
{
	if (cond)	bright = 1;
	else		bright = -1;
}
void MemoryEfficientObjectnessFilter::ScaleObjectnessMeasureOff() { scaleObjectnessMeasure = false; }
void MemoryEfficientObjectnessFilter::ScaleObjectnessMeasureOn()  { scaleObjectnessMeasure = true; }

//
//
//
void MemoryEfficientObjectnessFilter::Update()
{
	typedef itk::SmoothingRecursiveGaussianImageFilter <ImageType> FilterType;
	FilterType::Pointer filter = FilterType::New();

	filter->SetInput( input_image );
	filter->SetSigma( sigma );
	filter->Update();
	output_image = filter->GetOutput();

	output_image->DisconnectPipeline();

	GenerateObjectnessImage();
}

MemoryEfficientObjectnessFilter::ImagePointerType MemoryEfficientObjectnessFilter::GetOutput()	{ return output_image; }

void MemoryEfficientObjectnessFilter::GenerateObjectnessImage()
{
	// define variables for image size
	int w,h,d,wh,whd;
	w = output_image->GetLargestPossibleRegion().GetSize()[0];
	h = output_image->GetLargestPossibleRegion().GetSize()[1];
	d = output_image->GetLargestPossibleRegion().GetSize()[2];
	wh = w*h;
	whd = wh*d;

	//variables for browsing through image
	ImageType::IndexType image_index;	image_index[0]=0;image_index[1]=0;image_index[2]=0;
	int add;
	int pi, mi, pj, mj, pk, mk, p2i, m2i, p2j, m2j, p2k, m2k;

	//
	PixelType *img; //img = (PixelType *) calloc( wh*d, sizeof(PixelType) );
	img = output_image->GetBufferPointer();
	float hxx, hyy, hzz, hxy, hxz, hyz;
	float tmp;
	//float *l; l = (float *)calloc(3,sizeof(float));
	PixelType *tmp_obj;		tmp_obj = (PixelType *) calloc( whd, sizeof(PixelType) );
	PixelType *tmp_sum;		tmp_sum = (PixelType *) calloc( whd, sizeof(PixelType) );

	float al1, al2, al3, sum; double mean_norm=0;
	float Rsheet, Rblob, Rtube, Rnoise;
	float alpha_sq = 2*alpha*alpha, beta_sq = 2*beta*beta, gamma_sq = 2*gamma*gamma;

    unsigned pixelsInRoi = 0;

	for (int k=0 ; k<d ; k++)
	{
	    image_index[2] = k;
		pk=wh; p2k=2*wh; mk=-wh; m2k=-2*wh;
		if ( (k<2) || (k>d-3) )
		{
			if (k==0)	{mk=0; m2k=0;}
			if (k==d-1) {pk=0; p2k=0;}
			if (k==1)	m2k=-wh;
			if (k==d-2) p2k= wh;
		}

		for (int j=0 ; j<h; j++)
		{
            image_index[1] = j;
			pj=w; p2j=2*w; mj=-w; m2j=-2*w;
			if ( (j<2) || (j>h-3) )
			{
				if (j==0)	{mj=0; m2j=0;}
				if (j==h-1) {pj=0; p2j=0;}
				if (j==1)	m2j=-w;
				if (j==h-2) p2j= w;
			}
			for (int i=0 ; i<w ; i++)
			{
                image_index[0] = i;
				add = i + j*w + k*wh; //current pixel

                // dont process pixels outside roi
                if (roi_image.IsNotNull() && roi_image->GetPixel(image_index) == 0) {
                    tmp_obj[add] = 0;
                    continue;
                }

                pixelsInRoi++;

				pi=1; p2i=2; mi=-1;	m2i=-2;
				if ( (i<2) || (i>w-3) )
				{
					if (i==0)	{mi=0; m2i=0;}
					if (i==w-1) {pi=0; p2i=0;}
					if (i==1)	m2i=-1;
					if (i==w-2) p2i= 1;
				}

				tmp = 2.0*img[add];

				hxx = (img[add+m2i] - tmp + img[add+p2i])/4.0;
				hyy = (img[add+m2j] - tmp + img[add+p2j])/4.0;
				hzz = (img[add+m2k] - tmp + img[add+p2k])/4.0;
				hxy = (img[add+mi+mj] - img[add+pi+mj] - img[add+mi+pj] + img[add+pi+pj])/4.0;
				hxz = (img[add+mi+mk] - img[add+pi+mk] - img[add+mi+pk] + img[add+pi+pk])/4.0;
				hyz = (img[add+mj+mk] - img[add+pj+mk] - img[add+mj+pk] + img[add+pj+pk])/4.0;


                VectorType eigenVals;

                if (vector_image.IsNotNull()) {
                    VectorType principalEigenVector;
                    solve_3x3_symmetric_eigensystem(
                        hxx, hxy, hxz, hyy, hyz, hzz,
                        eigenVals, principalEigenVector);
                    vector_image->SetPixel(image_index, principalEigenVector);
                } else {
                    Eigenvalues_3_3_symetric(hxx, hxy, hxz, hyy, hyz, hzz, eigenVals);
                }


				al1 = fabs(eigenVals[0]); al2 = fabs(eigenVals[1]); al3 = fabs(eigenVals[2]);
				sum = al1+al2+al3;
				mean_norm+=sum;
				tmp_sum[add]=sum;

				if (al3==0)
				{
					tmp_obj[add] = 0;
				}
				else
				{
					Rtube  = al1 / (al2*al3);
					Rsheet = al2 / al3;
					Rblob  = 3.0*al1 / sum;


					if (objectDimension==1)
					{//Vesselness
						tmp_obj[add] = bright * (-eigenVals[2]/al3) * (1-exp(-Rsheet*Rsheet/alpha_sq)) * exp(-Rtube*Rtube/beta_sq) * exp(-Rblob*Rblob/gamma_sq);
					}
					else
					{//Sheetness
						tmp_obj[add] = bright * (-eigenVals[2]/al3) * exp(-Rsheet*Rsheet/alpha_sq) * exp(-Rtube*Rtube/beta_sq) * exp(-Rblob*Rblob/gamma_sq);
					}
					if (scaleObjectnessMeasure)	 tmp_obj[add] *= al3;
				}
			}
		}
	}
	mean_norm /= (float)pixelsInRoi;

    log("Mean norm = %1%") % mean_norm;

	for (int k=0 ; k<d ; k++)
	{
		image_index[2] = k;
		for (int j=0 ; j<h; j++)
		{
			image_index[1] = j;
			for (int i=0 ; i<w ; i++)
			{
				image_index[0] = i;
				add = i + j*w + k*wh;
				Rnoise = tmp_sum[add]/mean_norm;
				tmp_obj[add] *= (1 - exp(-Rnoise*Rnoise/0.25));
				output_image->SetPixel(image_index, tmp_obj[add]);
			}
		}
	}
	free(tmp_sum); 	free(tmp_obj);
}


//sorted by increasing absolute value
void MemoryEfficientObjectnessFilter::solve_3x3_symmetric_eigensystem(
    float M11, float M12, float M13, float M22, float M23, float M33,
    VectorType & eigenVals, VectorType & firstPrincipalEigenvector
) {

    // prepare
    float vals[9] = {
        M11, M12, M13,
        M12, M22, M23,
        M13, M23, M33
    };
    vnl_matrix<float> m(3,3,3*3, vals);

    // solve the eigensystem
    vnl_symmetric_eigensystem<float> eigenSolver(m);

    // store eigenvalues
    for (unsigned i=0; i<3; ++i) {
        eigenVals[i] = eigenSolver.get_eigenvalue(i);
    }

    // sort eigenvalues by increasing absolute values
	if (abs(eigenVals[0])>abs(eigenVals[1])) {swap(eigenVals[0],eigenVals[1]); }
	if (abs(eigenVals[1])>abs(eigenVals[2])) {swap(eigenVals[1],eigenVals[2]); }
	if (abs(eigenVals[0])>abs(eigenVals[1])) {swap(eigenVals[0],eigenVals[1]); }

    // find the first principal eigenvector
    for (unsigned i=0; i<3; ++i) {
        if (abs(eigenSolver.get_eigenvalue(i)) == abs(eigenVals[2])) {

            for (unsigned idx=0; idx < 3; ++idx)
                firstPrincipalEigenvector[idx] = eigenSolver.get_eigenvector(i)[idx];

            break;
        }
    }
}



//
//
//
//sorted by increasing absolute value ;
//ALGO FROM WIKIPEDIA, for finding the eigenvalues, but not the eigenvectors...
void MemoryEfficientObjectnessFilter::Eigenvalues_3_3_symetric(
    float M11, float M12, float M13, float M22, float M23, float M33,
    VectorType & eigenVals
) {
	double a, b, c, d; double t12, t13, t23, s23;
	a = -1.0;
	b = M11 + M22 + M33;
	t12 = M12*M12;	t13 = M13*M13; t23 = M23*M23; s23=M22*M33;
	c = t12 + t13 + t23 - M11*(M22+M33) - s23;
	d = M11*(s23 - t23) - M33*t12 + 2.0*M12*M13*M23 - M22*t13;

	double x,y,z;
	x = ( (3.0*c/-1.0) - (b*b)/1.0 ) / 3.0;							//x = ( (3.0*c/a) - (b*b)/(a*a) ) / 3.0;
	y = ((2.0*b*b*b/(-1.0)) - (9.0*b*c/1.0) + (27.0*d/a))/27.0;	//y = ((2.0*b*b*b/(a*a*a)) - (9.0*b*c/(a*a)) + (27.0*d/a))/27.0;
	z = y*y/4.0+x*x*x/27.0;

	double i, j, k, m, n, p;
	i = sqrt(y*y/4.0-z);
	j = -pow(i,1.0/3.0);
	k = acos(-y/(2.0*i));
	m = cos(k/3.0);
	n = sqrt(3.0)*sin(k/3.0);
	p = -(b/(3.0*a));

	double l1, l2, l3, tmp_flt;

	l1 = -2.0*j*m + p;	if (! (abs(l1) < std::numeric_limits<double>::max() )) {l1 = 0;}
	l2 = j*(m + n) + p;	if (! (abs(l2) < std::numeric_limits<double>::max() )) {l2 = 0;}
	l3 = j*(m - n) + p;	if (! (abs(l3) < std::numeric_limits<double>::max() )) {l3 = 0;}

	if (fabs(l1)>fabs(l2)) {tmp_flt=l1;l1=l2;l2=tmp_flt;}
	if (fabs(l2)>fabs(l3)) {tmp_flt=l2;l2=l3;l3=tmp_flt;}
	if (fabs(l1)>fabs(l2)) {tmp_flt=l1;l1=l2;l2=tmp_flt;}

	eigenVals[0] = (float)l1;
	eigenVals[1] = (float)l2;
	eigenVals[2] = (float)l3;
}






