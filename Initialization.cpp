#include "Initialization.h"
#include <thread>
#include <iostream>
#include "tracking.h"
//return the type of opencv paraments.
/*std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}*/

Initialization::Initialization(Tracking* tracking, const Frame& ReferenceFrame, int iterations)
{
  mvKeys1=ReferenceFrame.keypoints;
  mK=tracking->mK;
  MaxIterations=iterations;
}


bool Initialization::Initialize(const Frame& CurrentFrame, std::map<int,int> MatchedPoints, cv::Mat& R21, cv::Mat& t21,std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated)
{
  mvKeys2=CurrentFrame.keypoints;
  
  mMatchedKeys1.clear(); mMatchedKeys2.clear();
  for(int i=0;i<mvKeys1.size();i++)
  {
    if(MatchedPoints.count(i))
    {
      mMatchedKeys1.push_back(mvKeys1[i].pt);
      mMatchedKeys2.push_back(mvKeys2[MatchedPoints[i]].pt);
    }
  }
    
    float SH, SE;
    cv::Mat H,E;
     std::thread threadH(&Initialization::FindHomography,this,std::ref(SH), std::ref(H));
     std::thread threadE(&Initialization::FindEssentialMat,this,std::ref(SE), std::ref(E));
     threadH.join();
     threadE.join();
    // Compute ratio of scores
     float SCORE=SH/(SH+SE);
      if(SCORE>0.4){return RecoverPoseH(H,R21,t21,vP3D,vbTriangulated,1.0,50);}
      else return RecoverPoseE(E,R21,t21,vP3D,vbTriangulated,1.0,50);

}


//Compute Homography between source and destination planes. mvKeys1=H21*mvKeys2.
void Initialization::FindHomography(float& score,cv::Mat& H21)
{
  H21=cv::findHomography(mMatchedKeys1,mMatchedKeys2,CV_RANSAC,5.991,InlierH);
  score=cv::countNonZero(InlierH);
  //std::cout<<"Homography is: "<<H21<<" type: "<<type2str(H21.type())<<std::endl;
  std::cout<<"The number of the inlier points using FindHomography is: "<<score<<std::endl;
}

//Compute Essential Mat from the corresponding points in two images. mvKeys2*E21*mvKeys1=0.
void Initialization::FindEssentialMat(float& score, cv::Mat& E21)
{
  cv::Mat fundamental_matrix=cv::findFundamentalMat(mMatchedKeys1, mMatchedKeys2, CV_FM_RANSAC, 3.841, 0.99, InlierE);
  score=cv::countNonZero(InlierE);
  std::cout<<"The number of the inlier points using FindEssentialMat is: "<<score<<std::endl;
  //std::cout<<"fundamental_matrix is: "<<fundamental_matrix<<std::endl;
  E21=mK.t()*fundamental_matrix*mK;
  //std::cout<<"Essential Mat is: "<<E21<<std::endl;
}

//Recover pose(R21 t21)  and structure(vP3D vbTriangulated) from Homography.
bool Initialization::RecoverPoseH(cv::Mat Homography, cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated, double minParallax, int minTriangulated)
{
  std::cout<<"Initialize pose with recovery from Homography..."<<std::endl;
  
  // We recover 8 motion hypotheses using the method of Faugeras et al.
  // Motion and structure from motion in a piecewise planar environment.
  // International Journal of Pattern Recognition and Artificial Intelligence, 1988
  
  cv::Mat invK = mK.inv();
  cv::Mat A = invK*Homography*mK;
      
  cv::Mat U,w,Vt,V;
  cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
  V=Vt.t();

  float s = cv::determinant(U)*cv::determinant(Vt);
  
  double d1 = w.at<double>(0);
  double d2 = w.at<double>(1);
  double d3 = w.at<double>(2);

   if(d1/d2<1.00001 || d2/d3<1.00001)
   {
     return false;
   }
  
  //8 motion hypotheses.
  std::vector<cv::Mat> vR, vt, vn;
  vR.reserve(8);
  vt.reserve(8);
  vn.reserve(8);

  //for every d', n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
  double aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
  double aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
  double x1[] = {aux1,aux1,-aux1,-aux1};
  double x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
  double aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);
  double ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
  double stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
  for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_64F);
        Rp.at<double>(0,0)=ctheta;
        Rp.at<double>(0,2)=-stheta[i];
        Rp.at<double>(2,0)=stheta[i];
        Rp.at<double>(2,2)=ctheta;

	    cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_64F);
        tp.at<double>(0)=x1[i];
        tp.at<double>(1)=0;
        tp.at<double>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_64F);
        np.at<double>(0)=x1[i];
        np.at<double>(1)=0;
        np.at<double>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<double>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
  double aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
  double cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
  double sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
  for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_64F);
        Rp.at<double>(0,0)=cphi;
        Rp.at<double>(0,2)=sphi[i];
        Rp.at<double>(1,1)=-1;
        Rp.at<double>(2,0)=sphi[i];
        Rp.at<double>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_64F);
        tp.at<double>(0)=x1[i];
        tp.at<double>(1)=0;
        tp.at<double>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_64F);
        np.at<double>(0)=x1[i];
        np.at<double>(1)=0;
        np.at<double>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<double>(2)<0)
            n=-n;
        vn.push_back(n);
    }
    
    // Choose 1 motion by checking in triangulated points and parallax.
    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    double bestParallax = -1;
    std::vector<cv::Point3d> bestP3D;
    std::vector<bool> bestTriangulated;
    for (int i=0; i<8;i++)
    {
      std::vector<cv::Point3d> mvP3D;
      std::vector<bool> mvTriangulated;
      double mvparallax;
      int nGood=CheckRT(vR[i], vt[i], InlierH, mvP3D, 4, mvTriangulated, mvparallax);
      //std::cout<<"nGood: "<<nGood<<" points."<<std::endl;
       if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = mvparallax;
            bestP3D = mvP3D;
            bestTriangulated = mvTriangulated;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }   
    }
    //std::cout<<"best Rt : "<<bestGood<<" points."<<std::endl;
    //std::cout<<"Second Rt "<<secondBestGood<<" points."<<std::endl;
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*cv::countNonZero(InlierH))
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;
	std::cout<<"Initialize with "<<bestGood<<" points."<<std::endl;
        return true;
    }
    return false; 
}

bool Initialization::RecoverPoseE(cv::Mat EssentialMat, cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated,double minParallax, int minTriangulated)
{
   std::cout<<"Initialize pose with recovery from EssentialMat..."<<std::endl;
 
    cv::Mat R1, R2, t;
    //decomposeEssentialMat(EssentialMat, R1, R2, t);
	DecomposeE(EssentialMat,R1,R2,t);

    //std::cout<<"R1="<<R1<<std::endl;
    //std::cout<<"R12="<<R2<<std::endl;
    //std::cout<<"t="<<t<<std::endl;
    cv::Mat t1=t;
    cv::Mat t2=-t;
    
     // Reconstruct with the 4 hyphoteses and check
    std::vector<cv::Point3d> vP3D1, vP3D2, vP3D3, vP3D4;
    std::vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    double parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,InlierE,vP3D1, 4, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,InlierE, vP3D2, 4, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,InlierE, vP3D3, 4,vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,InlierE, vP3D4, 4, vbTriangulated4, parallax4);
    
    std::cout<<"Number of inlier RT from E="<<nGood1<<", "<<nGood2<<", "<<nGood3<<", "<<nGood4<<std::endl;
    
    int maxGood = std::max(nGood1,std::max(nGood2,std::max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = std::max(static_cast<int>(0.9*cv::countNonZero(InlierE)),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
    
}


// Check R&t by computing triangulated points and its parallax. Return the number of visible 3Dpoints.
// Maximum allowed reprojection error to treat a point pair as an inlier: 2 pixel.
int Initialization::CheckRT(const cv::Mat R, const cv::Mat t, cv::Mat vbMatchesInliers, std::vector<cv::Point3d> &vP3D, double th2, std::vector<bool> &vbGood, double &parallax)
{
    // Calibration parameters
    const double fx = mK.at<double>(0,0);
    const double fy = mK.at<double>(1,1);
    const double cx = mK.at<double>(0,2);
    const double cy = mK.at<double>(1,2);

    vbGood = std::vector<bool>(mMatchedKeys1.size(),false);
    vP3D.resize(mMatchedKeys1.size());

   std::vector<double> vCosParallax;
    vCosParallax.reserve(mMatchedKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1=cv::Mat::zeros(3,4,CV_64F);
    mK.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_64F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_64F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = mK*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;
    
    for(int i=0; i<mMatchedKeys1.size();i++)
    {
        if(!vbMatchesInliers.at<uchar>(i))
            continue;

        cv::Mat p3dC1;

        Triangulate(mMatchedKeys1[i],mMatchedKeys2[i],P1,P2,p3dC1);

        if(!std::isfinite(p3dC1.at<double>(0)) || !std::isfinite(p3dC1.at<double>(1)) || !std::isfinite(p3dC1.at<double>(2)))
        {
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        double dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        double dist2 = cv::norm(normal2);

        double cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<double>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<double>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        double im1x, im1y;
        double invZ1 = 1.0/p3dC1.at<double>(2);
        im1x = fx*p3dC1.at<double>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<double>(1)*invZ1+cy;

        double squareError1 = (im1x-mMatchedKeys1[i].x)*(im1x-mMatchedKeys1[i].x)+(im1y-mMatchedKeys1[i].y)*(im1y-mMatchedKeys1[i].y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        double im2x, im2y;
        double invZ2 = 1.0/p3dC2.at<double>(2);
        im2x = fx*p3dC2.at<double>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<double>(1)*invZ2+cy;

        double squareError2 =  (im2x-mMatchedKeys2[i].x)*(im2x-mMatchedKeys2[i].x)+(im2y-mMatchedKeys2[i].y)*(im2y-mMatchedKeys2[i].y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[i] = cv::Point3d(p3dC1.at<double>(0),p3dC1.at<double>(1),p3dC1.at<double>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[i]=true;
    }

    if(nGood>0)
    {
        std::sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = std::min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void Initialization::Triangulate(const cv::Point2f pt1, const cv::Point2f pt2, const cv::Mat P1, const cv::Mat P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_64F);

    A.row(0) = pt1.x*P1.row(2)-P1.row(0);
    A.row(1) = pt1.y*P1.row(2)-P1.row(1);
    A.row(2) = pt2.x*P2.row(2)-P2.row(0);
    A.row(3) = pt2.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<double>(3);
}

/*void Initialization::decomposeEssentialMat(cv::InputArray _E, cv::OutputArray _R1, cv::OutputArray _R2, cv::OutputArray _t)
{
  cv::Mat E = _E.getMat().reshape(1, 3);
  CV_Assert(E.cols == 3 && E.rows == 3);

  cv::Mat D, U, Vt;
  cv::SVD::compute(E, D, U, Vt);

  if (determinant(U) < 0) U *= -1.;
  if (determinant(Vt) < 0) Vt *= -1.;

  cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
  W.convertTo(W, E.type());

  cv::Mat R1, R2, t;
  R1 = U * W * Vt;
  R2 = U * W.t() * Vt;
  t = U.col(2) * 1.0;

  R1.copyTo(_R1);
  R2.copyTo(_R2);
  t.copyTo(_t);
}*/

void Initialization::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
	cv::Mat u,w,vt;
	cv::SVD::compute(E,w,u,vt);

	u.col(2).copyTo(t);
	t=t/cv::norm(t);

	cv::Mat W(3,3,CV_64F,cv::Scalar(0));
	W.at<double>(0,1)=-1;
	W.at<double>(1,0)=1;
	W.at<double>(2,2)=1;

	R1 = u*W*vt;
	if(cv::determinant(R1)<0)
		R1=-R1;

	R2 = u*W.t()*vt;
	if(cv::determinant(R2)<0)
		R2=-R2;
}
