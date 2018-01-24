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

def area_location_1(img):
  bbox = [0, 0, img.shape[1], img.shape[0]]
  binImg  = cv.blur(img, (3,3))

  #_,binImg = cv.threshold(binImg, 0, 255, cv.THRESH_OTSU)
  _,binImg = cv.threshold(binImg, 240, 255, cv.THRESH_BINARY)
  #cv.imwrite("temp.png", binImg)

  cv.imshow("", binImg)

  #_,contours,_ = cv.findContours(binImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  # Seems it will treat white as object, but my test image background is white and the object is black
  # so it will return the entire area os the while image is using RETR_EXTERNAL
  _,contours,_ = cv.findContours(binImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  print(len(contours))
  img = cv.drawContours(img, contours, -1, 0, 3)
  cv.imwrite("temp.png", img)
  cv.imshow("", img)
  cv.waitKey(0)

  '''
  for contour in contours:
    rect = cv.minAreaRect(contour)
    print(rect)
  '''
  return bbox

def get_lines(lines_in):
  if cv.__version__ < '3.0':
      return lines_in[0]
  return [l[0] for l in lines_in]

def area_location(img):
  bbox = [0, 0, img.shape[1], img.shape[0]]
  binImg  = cv.blur(img, (3,3))

  factor = 2.5
  cannyThreshold = 50
  canny_edges = cv.Canny(binImg, cannyThreshold, cannyThreshold*factor)
  lines = cv.HoughLinesP(canny_edges,rho=1,theta=np.pi/180,threshold=50,minLineLength=200,maxLineGap=15)
  print(lines.shape)
  #cv.imwrite("temp.png", binImg)

  '''
  while len(lines) > 30:
    cannyThreshold += 2
    canny_edges = cv.Canny(binImg, cannyThreshold, cannyThreshold*factor)
    lines = cv.HoughLines(canny_edges,1,np.pi/180,50,100,100)
  '''

  result = img.copy()
  for x1,y1,x2,y2 in get_lines(lines):    
    cv.line(result,(x1,y1),(x2,y2),(0,255,0),2) 

  '''
  for line in lines[0]:
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
  area_location(srcImg)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', type=str, default = 'res/1.jpg', help='the input image')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

