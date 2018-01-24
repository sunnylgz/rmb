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

# Reference
# http://blog.csdn.net/m0_38025293/article/details/73350507
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

NAME = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z']

area_fname = "res/temp.jpg"
def clip_roi(filename):
  area_size = np.array([130, 35])
  start_pos = np.array([20, 223])
  end_pos = area_size + start_pos
  srcImg = cv.imread(filename, 0)
  templateImg = cv.imread("res/template.jpg", 0)
  srcImg = cv.resize(srcImg, (templateImg.shape[1], templateImg.shape[0]), interpolation=cv.INTER_AREA)

  split = srcImg[start_pos[1]:end_pos[1], start_pos[0]:end_pos[0]]
  cv.imwrite(area_fname, split)


def horizontalProjectionMat(srcImg):
  binImg  = cv.blur(srcImg, (3,3))
  _,binImg = cv.threshold(binImg, 0, 255, cv.THRESH_OTSU)

  width = srcImg.shape[1]
  height = srcImg.shape[0]
  pixelMap = binImg == 0
  projectValArray = pixelMap.sum(axis=1)

  roiList = []
  inBlock = False
  for h in range(height):
    if not inBlock and projectValArray[h] != 0:
      inBlock = True
      startIndex = h
    elif inBlock and projectValArray[h] == 0:
      endIndex = h
      inBlock = False
      roiImg = srcImg[startIndex:endIndex, :]
      roiList.append(roiImg)

  return roiList

def verticalProjectionMat(srcImg):
  binImg  = cv.blur(srcImg, (3,3))
  _,binImg = cv.threshold(binImg, 0, 255, cv.THRESH_OTSU)

  width = srcImg.shape[1]
  height = srcImg.shape[0]
  pixelMap = binImg == 0
  projectValArray = pixelMap.sum(axis=0)

  roiList = []
  inBlock = False
  for w in range(width):
    if not inBlock and projectValArray[w] != 0:
      inBlock = True
      startIndex = w
    elif inBlock and projectValArray[w] == 0:
      endIndex = w
      inBlock = False
      roiImg = srcImg[:,startIndex:endIndex]
      roiList.append(roiImg)

  return roiList

def find_match_template(srcImg, isLetter=True, isNumber=True):
  starti = 0
  endi = 36
  if not isNumber:
    starti = 10
  if not isLetter:
    endi = 10

  binImg  = cv.blur(srcImg, (3,3))
  _,binImg = cv.threshold(binImg, 0, 255, cv.THRESH_OTSU)

  max_cnt = 0
  match = 0
  for j in range(starti, endi):
    nszName = "res/picture/%d.jpg" % (j)
    templateImg = cv.imread(nszName, 0)
    tempImg  = cv.blur(templateImg, (3,3))
    _,tempImg = cv.threshold(tempImg, 0, 255, cv.THRESH_OTSU)

    compare = binImg == tempImg
    count = compare.sum()
    #count = np.sum(np.logical_and(np.logical_not(binImg), np.logical_not(tempImg)))

    if max_cnt < count:
      max_cnt = count
      match = j

  return NAME[match], max_cnt

def recognize_number():
  srcImg = cv.imread(area_fname, 0)
  b = verticalProjectionMat(srcImg)

  for i in range(len(b)):
    a = horizontalProjectionMat(b[i])
    szName = "res/split/%d.jpg" % (i)

    for j in range(len(a)):
      templateImg = cv.imread("res/picture/2.jpg", 0)
      split = cv.resize(a[j], (templateImg.shape[1], templateImg.shape[0]), interpolation=cv.INTER_AREA)
      cv.imwrite(szName, split)

  recognized_name = []
  szName = "res/split/0.jpg"
  srcImg = cv.imread(szName, 0)
  name, count = find_match_template(srcImg, isNumber = False)
  recognized_name.append(name)

  match_count = []
  letter_index = []
  for i in range(1,4):
    szName = "res/split/%d.jpg" % (i)
    srcImg = cv.imread(szName, 0)
    name, count = find_match_template(srcImg)
    if name >= 'A':
      letter_index.append(i)
      match_count.append(count)

    recognized_name.append(name)

  if len(letter_index) > 1:
    max_matched = match_count.index(max(match_count))
    del letter_index[max_matched]
    for i in letter_index:
      szName = "res/split/%d.jpg" % (i)
      srcImg = cv.imread(szName, 0)
      name, _ = find_match_template(srcImg, isLetter = False)
      recognized_name[i] = name
  elif len(letter_index) == 0:
    max_count = 0
    for i in range(1,4):
      szName = "res/split/%d.jpg" % (i)
      srcImg = cv.imread(szName, 0)
      name, count = find_match_template(srcImg, isNumber = False)
      if count > max_count:
        max_count = count
        letter_index = i
        letter = name

    recognized_name[letter_index] = letter

  for i in range(4,10):
    szName = "res/split/%d.jpg" % (i)
    srcImg = cv.imread(szName, 0)
    name, _ = find_match_template(srcImg, isLetter = False)
    recognized_name.append(name)

  print(recognized_name)

def main(args):
  clip_roi(args.input)
  recognize_number()

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
