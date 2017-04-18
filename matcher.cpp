#include "matcher.h"
#include <stdint.h>
#include "KeyFrame.h"
int Matcher::SearchForInitialization(Frame* fr1, Frame* fr2, int radius)
{
	int Matched_num = 0;
	for (int i = 0; i < fr1->keypoints.size(); i++)
	{
		cv::KeyPoint kp1 = fr1->keypoints.at(i);
		int x = kp1.pt.x;
		int y = kp1.pt.y;
		int min_SSD_error = SSD_error_th;  int Matched_id = -1;
		for (int j = 0; j < fr2->keypoints.size(); j++)
		{
			cv::KeyPoint kp2 = fr2->keypoints.at(j);
			if ((kp2.pt.x - x)*(kp2.pt.x - x) + (kp2.pt.y - y)*(kp2.pt.y - y) < radius*radius)
			{
				int SSD_error = SSDcompute(fr1, fr2, kp1, kp2);
				if (SSD_error < min_SSD_error)
				{
					min_SSD_error = SSD_error;
					Matched_id = j;
				}
			}
		}
		if (Matched_id > -1){
			MatchedPoints[i] = Matched_id;
			Matched_num++;
		}
	}
	return Matched_num;
}

//compute the sum of squared difference (SSD) between patches(5x5).
int Matcher::SSDcompute(Frame* fr1, Frame* fr2, cv::KeyPoint kp1, cv::KeyPoint kp2)
{
	int x1 = kp1.pt.x; int y1 = kp1.pt.y;
	int x2 = kp2.pt.x; int y2 = kp2.pt.y;
	cv::Mat fr1_Range = fr1->image.colRange(x1 - 2, x1 + 3).rowRange(y1 - 2, y1 + 3);
	cv::Mat fr2_Range = fr1->image.colRange(x2 - 2, x2 + 3).rowRange(y2 - 2, y2 + 3);
	cv::Mat differ = fr1_Range - fr2_Range;
	return differ.dot(differ);
}

int Matcher::SearchMatchByGlobal(Frame* fr1, std::map<KeyFrame*, cv::Mat> globalH)
{
	std::unordered_multimap<cv::KeyPoint*, std::pair<Frame*, cv::KeyPoint*>> data;
	for (auto i = globalH.begin();i!=globalH.end(); i++)
	{
		auto tmpd = matchByH(i->first, fr1, i->second);
		data.insert(tmpd.begin(), tmpd.end());
	}
	fr1->matchedGroup = data;
	return data.size();
}

// Find corresponding feature points in two frames with homography
// fr1 represent keyframe to project.
// TODO: different search range for well and ill conditioned points
std::unordered_multimap<cv::KeyPoint*, std::pair<Frame*, cv::KeyPoint*>> Matcher::matchByH(Frame* fr1, Frame* fr2, cv::Mat H)
{
	cv::Mat_<double> kpl(3,1);
	cv::Mat_<double> ppl(3, 1);
	std::unordered_multimap<cv::KeyPoint*, std::pair<Frame*, cv::KeyPoint*>> ret;
	int width = fr1->image.size().width;
	int height = fr1->image.size().height;
	for (int i = 0; i < fr1->keypoints.size(); i++)
	{
		cv::KeyPoint kp = fr1->keypoints[i];
		kpl(0) = kp.pt.x; kpl(1) = kp.pt.y; kpl(2) = 1;
		cv::Mat_<uint8_t> warped(patchHalfSize * 2, patchHalfSize * 2, uint8_t(0));
		//calculate warped patch.
		for (int ii = -patchHalfSize; ii < patchHalfSize; ii++)
		{
			for (int jj = -patchHalfSize; jj < patchHalfSize; jj++)
			{
				// equation 7 in rkslam.
				kpl = H*kpl;
				kpl(0) = kpl(0) + ii; kpl(1) = kpl(1) + jj;
				kpl = kpl / kpl(2);
				kpl = H.inv()*kpl;
				kpl = kpl / kpl(2);
				if (kpl(0)>0 && kpl(0) < width&&kpl(1) > 0 && kpl(1) < height)
				{
					warped(jj + patchHalfSize, ii + patchHalfSize) = fr1->image.at<uint8_t>(kpl(1), kpl(0));
				}
			}
		}
		int min_SSD_error = SSD_error_th;
		cv::KeyPoint* matchedKp = nullptr;
		// iterate to find the smallest error in fr2.
		for (int j = 0; j < fr2->keypoints.size(); j++)
		{
			cv::Point2f pt2 = fr2->keypoints[j].pt;
			if (abs(ppl(0) - pt2.x)<globalSearchHalfLength&&abs(ppl(1)-pt2.y)<globalSearchHalfLength)
			{
				int u = pt2.y-patchHalfSize;int d = pt2.y+patchHalfSize;int l = pt2.x-patchHalfSize;int r = pt2.x+patchHalfSize;
				if (u<0||d>=height||l<0||r>=width)
				{
					continue;
				}
				cv::Mat patch = fr2->image.rowRange(pt2.y - patchHalfSize, pt2.y + patchHalfSize).colRange(pt2.x - patchHalfSize, pt2.x + patchHalfSize);
				cv::Mat differ = (warped - patch)&(warped != 0);
				int ssdError = differ.dot(differ);
				if (ssdError < min_SSD_error)
				{
					matchedKp = &(fr2->keypoints[j]);
					min_SSD_error = ssdError;
				}
			}
		}
		if (matchedKp != nullptr)
		{
			auto p = std::make_pair(static_cast<KeyFrame*>(fr1), matchedKp);
			ret.insert(std::make_pair(&kp, p));
		}
	}
	return ret;
}

inline cv::Mat warpPoint(cv::Mat_<double> point, cv::Mat H)
{
	return H*point / point(2);
}
cv::Mat Matcher::warpPatch(Frame* fr1, cv::KeyPoint kp1, cv::Mat H)
{
	double width = fr1->image.size().width;
	double height = fr1->image.size().height;
	cv::Mat_<double> upleft(2, 1);
	cv::Mat_<double> downleft(2, 1);
	cv::Mat_<double> upright(2, 1);
	cv::Mat_<double> downright(2, 1);
	int l = kp1.pt.x - patchHalfSize >= 0 ? kp1.pt.x - patchHalfSize : 0;
	int r = kp1.pt.x + patchHalfSize <= width - 1? kp1.pt.x + patchHalfSize : width-1;
	int u = kp1.pt.y - patchHalfSize >= 0 ? kp1.pt.y - patchHalfSize : 0;
	int d = kp1.pt.y + patchHalfSize <= height - 1 ? kp1.pt.y + patchHalfSize : height-1;

	cv::Mat roi = fr1->image.colRange(l, r+1).rowRange(u, d+1);
	upleft(0) = l; downleft(0) = l; upright(0) = r; downright(0) = r;
	upleft(1) = u; downleft(1) = d; upright(1) = u; downright(1) = d;
	upleft(2) = downleft(2) = upright(2) = downright(2) = 1;
	//transform corner points to predict warped image patch size;
	upleft = warpPoint(upleft, H); upright = warpPoint(upright, H);
	downleft = warpPoint(downleft, H); downright = warpPoint(downright, H);
	double ll = MIN(upleft(0), downleft(0));
	double rr = MAX(upright(0), downright(0));
	double uu = MIN(upleft(1), upright(1));
	double dd = MAX(downleft(1), downright(1));
	cv::Mat warped(dd - uu, rr - ll, CV_8UC1);
	cv::warpPerspective(roi, warped, H, cv::Size(rr - ll, dd - uu));
	return warped;
}

