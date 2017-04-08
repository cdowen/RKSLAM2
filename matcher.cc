#include "matcher.h"

int matcher::SearchForInitialization(Frame* fr1, Frame* fr2, int radius)
{
  int Matched_num=0;
  for(int i=0; i<fr1->keypoints.size(); i++)
  {
    cv::KeyPoint kp1=fr1->keypoints.at(i);
   int x=kp1.pt.x;
   int y=kp1.pt.y;
   int min_SSD_error=SSD_error_th;  int Matched_id=-1;
    for(int j=0; j<fr2->keypoints.size();j++)
    {
      cv::KeyPoint kp2=fr2->keypoints.at(j);
      if((kp2.pt.x-x)*(kp2.pt.x-x)+(kp2.pt.y-y)*(kp2.pt.y-y)<radius*radius)
      {
	int SSD_error=SSDcompute(fr1,fr2,kp1,kp2);
	if (SSD_error<min_SSD_error)
	{
	  min_SSD_error=SSD_error;
	  Matched_id=j;
	}
      }
    }
    if(Matched_id>-1){
      MatchedPoints[i]=Matched_id;
      Matched_num++;
    }
  }
  return Matched_num;
}

//compute the sum of squared difference (SSD) between patches(5x5).
int matcher::SSDcompute(Frame* fr1, Frame* fr2, cv::KeyPoint kp1, cv::KeyPoint kp2)
{
   int x1=kp1.pt.x; int y1=kp1.pt.y;
   int x2=kp2.pt.x; int y2=kp2.pt.y;
   cv::Mat fr1_Range=fr1->image.colRange(x1-2,x1+3).rowRange(y1-2,y1+3);
   cv::Mat fr2_Range=fr1->image.colRange(x2-2,x2+3).rowRange(y2-2,y2+3);
   cv::Mat differ=fr1_Range-fr2_Range;
   return differ.dot(differ);
}

