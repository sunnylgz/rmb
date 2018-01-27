#! /usr/bin/env python3
"""
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Reference:
# https://wenku.baidu.com/view/671d04385727a5e9856a615d.html?mark_pay_doc=2&mark_rec_page=1&mark_rec_position=3&clear_uda_param=1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
from scipy import misc
import pickle
import numpy as np
import cv2 as cv
import datetime,time
import math

def find_cross_point(line1, line2):
  x1,y1,x2,y2 = line1
  x3,y3,x4,y4 = line2

  # for vertical lines, there's no cross point
  # '1' is experiental number
  if abs((x1-x2)/(y1-y2)) + abs((x3-x4)/(y3-y4))  < 1:
    return None

  d = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)
  ad = d / ((x1-x2)*(x3-x4))
  print(ad)
  # for parallel lines, there's no cross point
  # '2.5' is experiental number
  if abs(ad) > 2.5: #d != 0:
    x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) // d
    y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) // d

    return [x, y]

  return None

# TODO: not completed
def find_vertex(contour, minAreaBox):
  approxCurve	=	cv.approxPolyDP(contour, 3, True)
  print("approx curve shape: ", approxCurve.shape)

  candidates = approxCurve
  c_left = [candidates[0,0]]
  c_right = [candidates[0,0]]
  c_bottom = [candidates[0,0]]
  c_top = [candidates[0,0]]
  candidates = candidates[1:,:]

  flags = [False, False, False, False] # topleft, topright, bottomright, bottomleft

  for candidate in candidates:
    if candidate[0,0] < c_left[0][0]:
      c_left.clear()
      c_left.append(candidate[0])
    elif candidate[0,0] == c_left[0][0]:
      c_left.append(candidate[0])

    if candidate[0,1] < c_top[0][1]:
      c_top.clear()
      c_top.append(candidate[0])
    elif candidate[0,1] == c_top[0][1]:
      c_top.append(candidate[0])

    if candidate[0,0] > c_right[0][0]:
      c_right.clear()
      c_right.append(candidate[0])
    elif candidate[0,0] == c_right[0][0]:
      c_right.append(candidate[0])

    if candidate[0,1] > c_bottom[0][1]:
      c_bottom.clear()
      c_bottom.append(candidate[0])
    elif candidate[0,1] == c_bottom[0][1]:
      c_bottom.append(candidate[0])


  print("left: ", c_left)
  print("right: ", c_right)
  print("top: ", c_top)
  print("bottom: ", c_bottom)

  return approxCurve

def points_distance(point1, point2):
  x1, y1 = point1
  x2, y2 = point2

  return math.sqrt((y2-y1)**2 + (x2-x1)**2)
def vertex_point_filter(points, w, h):
  minDistance = 20
  i = 0
  while i < len(points):
    if points[i][0] < -10 or points[i][0] > w+10 or points[i][1] < -10 or points[i][1] > h+10:
      del points[i]
      continue
    points[i][0] = max(0, points[i][0])
    points[i][1] = max(0, points[i][1])
    points[i][0] = min(w-1, points[i][0])
    points[i][1] = min(h-1, points[i][1])
    i += 1
  for i in range(len(points)):
    j = i+1
    while j < len(points):
      if points_distance(points[i], points[j]) < minDistance:
        del points[j]
      else:
        j += 1
  return points

def area_location(img, bg_black=True):
  bbox = [0, 0, img.shape[1], img.shape[0]]
  min_area = img.shape[1]*img.shape[0] * 0.1
  binImg  = cv.blur(img, (3,3))

  if bg_black:
    ThresholdVal = 100
  else:
    ThresholdVal = 240
  #_,binImg = cv.threshold(binImg, 0, 255, cv.THRESH_OTSU)
  _,binImg = cv.threshold(binImg, ThresholdVal, 255, cv.THRESH_BINARY)

  if not bg_black:
    binImg = 255 - binImg
  cv.imwrite("binary.png", binImg)

  _,contours,_ = cv.findContours(binImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  # Seems it will treat white as object, but my test image background is white and the object is black
  # so it will return the entire area os the while image is using RETR_EXTERNAL
  #_,contours,_ = cv.findContours(binImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  box_candidate = []
  contours_candidate = []
  for contour in contours:
    rect = cv.minAreaRect(contour)
    print(rect)
    box = cv.boxPoints(rect)
    box =np.int0(box)
    print(box)
    (box_w,box_h)=np.int0(rect[1])
    print(box_w,box_h)

    if (box_w * box_h < min_area):
      continue
    angle = rect[2]
    if (box_h > box_w):
      angle += 90

    print("angle is ", angle)
    vertex = find_vertex(contour, box)
    box_candidate.append(box)
    contours_candidate.append(contour)
    print("contour shape: ", contour.shape)

  color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
  color_img = cv.drawContours(color_img, contours_candidate, -1, (0, 0, 255), 3)
  color_img = cv.drawContours(color_img, box_candidate, -1, (0, 255, 0), 3)
  for v in vertex:
    cv.circle(color_img, (v[0,0], v[0,1]), 2, (255, 0, 0))

  tempimg = np.zeros(img.shape, dtype=np.uint8)
  tempimg = cv.drawContours(tempimg, contours_candidate, -1, 255)
  lines = cv.HoughLinesP(tempimg,rho=1,theta=np.pi/180,threshold=50,minLineLength=min(img.shape)/3,maxLineGap=100)
  print("find lines number: ", len(lines))
  tempimg = np.zeros(img.shape, dtype=np.uint8)
  lines = get_lines(lines)
  if len(lines) < 4:
    raise RuntimeError("Can't find the 4 sides of one object")

  #for x1,y1,x2,y2 in lines:
  #  cv.line(tempimg,(x1,y1),(x2,y2),(255,255,0),2)

  crossPnts = []
  for i in range(len(lines)):
    line1= lines[i]
    for j in range(i, len(lines)):
      line2 = lines[j]
      crossP = find_cross_point(line1, line2)
      if crossP != None:
        crossPnts.append(crossP)
        print("a vertex")
        '''
        tempimg = np.zeros(img.shape, dtype=np.uint8)
        x1,y1,x2,y2 = line1
        cv.line(tempimg,(x1,y1),(x2,y2),(255,255,0),2)
        cv.circle(tempimg, (crossP[0], crossP[1]), 2, (255, 0, 0))
        x1,y1,x2,y2 = line2
        cv.line(tempimg,(x1,y1),(x2,y2),(255,255,0),2)
        cv.imshow("", tempimg)
        cv.waitKey(0)
        '''

  print("find vertex", len(crossPnts), crossPnts)
  crossPnts = vertex_point_filter(crossPnts, img.shape[1], img.shape[0])
  print("after filtering", len(crossPnts), crossPnts)
  if len(crossPnts) < 4:
    raise RuntimeError("Can't find the 4 vertex of one object")
  for v in crossPnts:
    cv.circle(tempimg, (v[0], v[1]), 2, (255, 0, 0))

  # sort by the distance from (0,0)
  crossPnts = sorted(crossPnts, key=lambda point: points_distance((0,0), point))
  lefttop = crossPnts[0]
  rightbottom = crossPnts[-1]
  # TODO: need deal w/ len(crossPnts) > 4
  if crossPnts[1][1] < crossPnts[2][1]:
    righttop = crossPnts[1]
    leftbottom = crossPnts[2]
  else:
    righttop = crossPnts[2]
    leftbottom = crossPnts[1]
  cv.putText(tempimg, "lefttop", tuple(lefttop), cv.FONT_HERSHEY_PLAIN, 1, 255)
  cv.putText(tempimg, "righttop", tuple(righttop), cv.FONT_HERSHEY_PLAIN, 1, 255)
  cv.putText(tempimg, "leftbottom", tuple(leftbottom), cv.FONT_HERSHEY_PLAIN, 1, 255)
  cv.putText(tempimg, "rightbottom", tuple(rightbottom), cv.FONT_HERSHEY_PLAIN, 1, 255)

  #AffineMatrix = cv.getAffineTransform(np.float32([lefttop, righttop, rightbottom]),
  #                                    np.float32([[0,0], [img.shape[1], 0], [img.shape[1], img.shape[0]]]))
  #resultimg = cv.warpAffine(img, AffineMatrix, (img.shape[1], img.shape[0]))
  PerspectiveMatrix = cv.getPerspectiveTransform(np.float32([lefttop, righttop, leftbottom, rightbottom]),
                                      np.float32([[0,0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]]))
  resultimg = cv.warpPerspective(img, PerspectiveMatrix, (img.shape[1], img.shape[0]))

  cv.imwrite("contours.png", color_img)
  cv.imwrite("temp.png", tempimg)
  cv.imwrite("result.png", resultimg)
  cv.imshow("", resultimg)
  cv.waitKey(0)

  return bbox

def get_lines(lines_in):
  if cv.__version__ < '3.0':
      return lines_in[0]
  return [l[0] for l in lines_in]

# TODO: Canny can return good results, but HoughLinesP can't
def area_location_canny(img):
  bbox = [0, 0, img.shape[1], img.shape[0]]
  binImg  = cv.GaussianBlur(img, (3,3), 0)
  #binImg  = cv.blur(img, (3,3))

  factor = 2.5
  cannyThreshold = 50
  houghThreshold = 50
  canny_edges = cv.Canny(binImg, cannyThreshold, cannyThreshold*factor)
  lines = cv.HoughLinesP(canny_edges,rho=1,theta=np.pi/180,threshold=houghThreshold,minLineLength=100,maxLineGap=100)
  #lines = cv.HoughLines(canny_edges, 1, np.pi/180, 150, 0, 0)

  
  while len(lines) > 30:
    houghThreshold += 2
    canny_edges = cv.Canny(binImg, cannyThreshold, cannyThreshold*factor)
    lines = cv.HoughLinesP(canny_edges,rho=1,theta=np.pi/180,threshold=houghThreshold,minLineLength=100,maxLineGap=100)
  
  cv.imwrite("edges.png", canny_edges)
  print(lines.shape)
  result = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

  for x1,y1,x2,y2 in get_lines(lines):    
    cv.line(result,(x1,y1),(x2,y2),(0,0,255),2) 

  '''
  for line in get_lines(lines):
    print(line)
    rho = line[0]
    theta= line[1]
    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
      pt1 = (int(rho/np.cos(theta)),0)    
      pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])    
      cv.line( result, pt1, pt2, (255))    
    else:
      pt1 = (0,int(rho/np.sin(theta)))    
      pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))    
      cv.line(result, pt1, pt2, (0), 3)  
  '''
  cv.imwrite("lines.png", result)
  #cv.imshow("edges", canny_edges)
  cv.imshow("result", result)


  cv.waitKey(0)

  '''
  for contour in contours:
    rect = cv.minAreaRect(contour)
    print(rect)
  '''
  return bbox

def main(args):
  srcImg = cv.imread(args.input, 0)
  if args.white:
    area_location(srcImg)
  else:
    area_location(srcImg, False)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', type=str, default = 'res/1.jpg', help='the input image')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('-w', '--white', action="store_true", help='white backgroud')
    parser.add_argument('--classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

