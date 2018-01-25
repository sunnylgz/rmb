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
  cv.imwrite("contours.png", color_img)
  cv.imshow("", color_img)
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

