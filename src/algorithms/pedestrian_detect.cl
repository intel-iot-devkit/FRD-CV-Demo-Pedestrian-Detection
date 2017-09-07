// Copyright (C) 2013-2016 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

/**
 * @mainpage The Pedestrian detection reference design for Altera FPGAs.
 * The project derives from the OpenCV pedestrian detection implementation. This version demonstrates a hardware friendly coding style, while
 * remaining of comparable quality with the OpenCV reference.
 * @file pedestrian_detect.cl
 * @brief The kernel file for the pedestrian detection algorithm
 * @version 1.0
 * @section DESCRIPTION
 * The kernel implements the HOG algorithm using OpenCL 
 * The HOG (Histogram of Oriented Gradient) algorithm is based on a paper by Navneet Dalal and Bill Triggs "Histogram of oriented gradients for human detection" 2005.
 * The paper proposed four crucial steps for processing an image, including:
 * - 1. Gradient computation: It converts the color space to gradient space, by calculating each pixel's gradient using the standard formula.
 * - 2. Histogram computation: The gradient image is partitioned into a series of blocks, where histograms are created based on the gradients.
 * - 3. Normalization: Each block histogram is normalized in order to smoothen the accumulated difference, using L2-norm formula.
 * - 4. SVM computation: Using an SVM-based algorithm, the processed image is compared with a reference vector to determine the similarity of the analyzed data to a pedestrian.
 * In addition, in order to identify pedestrians of variable size, a resizing procedure wraps the core algorithm. The image is down sampled and pedestrian identification is done for each resized version of the original image.
 *
 * The reference source can be found in OpenCV library http://docs.opencv.org/modules/gpu/doc/object_detection.html.
 * The algorithm has been slightly restructured to adhere to hardware-friendly coding styles and achieve the highest performance. The most significant changes are listed below:
 * - 1. The algorithm has been organized to better expose the data flow. It is now split into 5 kernels. The host program can invoke the kernels concurrently, and ensure their synchronization.
 * - 2. The loops have been adapted such that the compiler can parallelize aggresivly the single-threaded code. 
 * - 3. Data type conversions: the floating point operations are selectively substituted by integer operation, in order to consume less FPGA resources. Meanwhile, it remains similar detection quality and user experience. 
 * - 4. Often, integer multiplications are truncated to 27-bits to match the witdh of the multipliers in the DSP blocks. The compiler determines how many multiplication bits to preserve, and masking the result (with an & operation) can indicate to the compiler that fewer bits are needed.
 * - 5. The memory accesses are refactored according to Altera's OpenCL Best Practices Guide. This fully takes advantage of on-chip channels and local shift register chains rather than global memory accessing, which improves both the performance and hardware efficiency.
 *
 * @see http://docs.opencv.org/modules/gpu/doc/object_detection.html
 * @see http://www.altera.com/devices/processor/soc-fpga/cyclone-v-soc/overview/cyclone-v-soc-overview.html
 * @see http://www.altera.com/literature/lit-opencl-sdk.jsp
*/
/* The data flow is illustrated in the following
 * ____________________________________________________________________________________________
 *  Host program            |       On-chip kernels
 *    (image data in        |
 *      RGB format   )  >>>>>>>>>>> [Global memory]
 *                          |         v
 *                          |      [Kernel] Resize()                  : Resize the image into fixed size
 *                          |         v
 *                          |      [Channel] chResize_Gradient
 *                          |         v
 *                          |      [Kernel] Gradient()                : Compute the gradient for each pixel
 *                          |         v  \---------------------------v
 *                          |      [Channel] chGrad_Hist_weight  [Channel] chGrad_Hist_angle
 *                          |         v  /---------------------------/ : Transmit the gradient values and angles
 *                          |         v  v
 *                          |      [Kernel] Histograms()              : Compute the histogram block by block
 *                          |         v
 *                          |      [Channel] chHist_Norm
 *                          |         v
 *                          |      [Kernel] Normalizeit()             : Normalize the blocks one by one
 *                          |         v 
 *                          |      [Global memory]
 *                          |         v
 *                          |      [Kernel] Svm()                     : Use SVM algorithm to process and analyze the blocks
 *                          |         v
 *   Show the image and <<<<<<<<?? [Global memory]
 *   highlight pedestrians  |
 *--------------------------------------------------------------------------------------------------
 */

#define WX 7     /*!< The block number per row for svm() */
#define WY 15    /*!< The block number per row for svm() */
#define MAXBX 105 /*!< The maximum number of blocks supported per row. It must be a multiple of WX !! It must cover the border around the image. */
#define MAXBX_SVM 99 /*!< The maximum number of blocks supported per row. It must be a multiple of WX !! It must cover the border around the image. */
#define BLOCK_SIZE 2 /*!< The number of cells per dimension in one block */
#define CELL_SIZE 8  /*!< The number of pixels per cell */
#define NBINS 9      /*!< The number of bins per cell */
#define MAXCOLS (MAXBX * CELL_SIZE)
#define BLOCK_HIST (NBINS * BLOCK_SIZE * BLOCK_SIZE)

#define AMP_Hist 4   /*!< The original histograms() data is scaled by 2^AMP_Hist */
#define AMP_Svm 6    /*!< The original svm() data is scaled by 2^AMP_Svm */
#define AMP_Norm 5   /*!< The original normalizeit() data is scaled by 2^AMP_Norm */
#define SHR_Inhist 11 /*!< The histograms() data is truncated by this amount */

channel uchar4  chResize_Gradient  __attribute__((depth(MAXCOLS)));          /*!< The channel from resize() to gradient() */
channel ushort2 chGrad_Hist_weight __attribute__((depth(MAXCOLS / 16 + 8))); /*!< The channel from gradient() to histograms() for weights */
channel uchar2  chGrad_Hist_angle  __attribute__((depth(MAXCOLS / 16 + 8))); /*!< The channel from gradient() to histograms() for angles */
channel ushort8 chHist_Norm        __attribute__((depth(MAXBX *5 + 10)));    /*!< The channel from histograms() to normalizeit() */
channel ushort  chNorm_Svm         __attribute__((depth(CELL_SIZE)));        /*!< The channel from normalizeit() to svm() */

 /*! @brief The constant detection vector, being times AMP_Matrix 128 but shift >>1 */
constant char detector[WY * WX ][BLOCK_HIST] = {
{  7,-19, -7,  6, 15, -5,  6, -7, 11, 13, -3,  1,  2, 14,  2, 11, -8, 17, 10, -7, -6, 12, 14,-10, -7, -6,  4,  6,  0, 10, 10, 10,  8, 12, -5,  4},
{  7, -6,  5, -5,  9, -5,  1,  2, 14, 11,  0, 14, 10, 13,  3, 15, 17, 13,  7, -9, -1,  1,  4,  2,  3,  3,  6,  9,  4, -8,  3,  8, 11,  1,  3,  3},
{  7, -3,  6,  4, 11,  6, 11, 10,  3, 13,  8,  8,  1,  7,  9, 11,  9,  8, -5, -1, -8,  4,  6,  9,  4,  0, -2, -4, -2, -4,  1, -5,  0,  7, 15, -3},
{  7,  8,  7, -1,  4,  7, -1, -1,  8, -3,  7,-11, -9,  6, -2, -3,  4, -6, -6,  4,  8,  3, -2,  1,  1,  7, -3, -6, -5,  1, -3, -3, -3, -4, -1, -8},
{  1, 12, -5, -9, 19, -7, -3,  7,  4, -9, -7,-13,-14, 10,-10,-11,  4,  3,  2, -2, -4, -3, 12,  6, 18, 15,  0, -6,  0,-12,-15,  6,  2,  0, 14,  0},
{-14,-10,-13, -2, 21, -1, -5, -1,  1, -2, -6,-16,  0, -2,-10, -3,  0,  3, -2, -5, -8,  4, 16,  8, -1, 14,  2,  3,  5, -7, 10,  8, -7,-13, 10, -3},
{  2,  0, -5,  9,  7, -4, -3,  4,  6, -7, -6, -6,  2,  3, -3,-18,  7,  2,  7,  6,  0, 10, 12,  1, -4, 14, -4, -9, -8, -3, -1, -2,  2,-13,  6, -7},
{ -3,  1,-11,  5,  8,  2,-10,  5, -4, -8, -5,-10,  0, -4, -2, -8,  6, -6, 11,  3,  4, 13,  9,  6, -6, 17, 11,  3,  0, -5, -3, -5, -2,  3, 15, -8},
{ -4, -6,  6,  8,  4,  2,  0,  6,  0, -5, -1,  1,  4, -4, -3, -2, -1, -3,  3, -2,  1, -2,  1,  6,  5,  9,  7, -5,  2, -1,  5,-10,  3, -5, -4, -5},
{  0, -6, -9,  1, -3,  0, -4, -4, -4, -2, -6,-12,-13, -5,  0, -9, -2, -5, -2,  2, -3, 12, -5,  7,  5,  2, -4,-16,  9, -5,  1,  1,  7,  1,  4, -7},
{ -5, -4, -6, -1,  2, -1,-12, -2, -3,-13,-10,  0, -1,  2, -8,-14, -3, -3, -9,  9,  0,  1,  1,  5, -4,  5, -6,-15,  4,  0, 14,  7,  0, -7,  1, -7},
{ -1, -7, -2,  2,  1, -1, -8,  0,  3, -3, -9, -8, -4,-11, -4, -9, -3, -8,  0, -7, -6, 10,  0,  5,  4,  7,  5, -4, -5, -4, -4,  2,  3, -3,  0,-13},
{ -6, -4,  0,  1,  1, -6, -6, -5, -9,-14,  0,  0, -8, -8,-11, -6,  0, -8,  0, -2,  5,  7,  9,  1,  8,  2, -1, -4,  1,  1,  0,  3, -4,  9, -1, -1},
{-10,  7,  8,  5,  0,-12, -5,  5,  2, -6, -1,  3,  6, 12,-10, -6, -7, -1, -7,  1, -5, -9,  5,  3,  4,  2,  7,  1, -4,  8,  3,  2, -7,-13,-11,  9},
{ -3, -3, -3, -6,  0,  7, -2, -2,  8,  6, -4,  1, -3, -7,  0, -4,  2, 15,  0, -8,  7,  0, -6, 12, -5, -7,  2,  3,-14, -1, -1,-13,  4, -3, -9,  7},
{ 14,-11, -4,  1, 19,  0, -7, -9,  7, 17, -6, 13,  6, 16, 12,  8,  2, 12,  8,-18, -4,  4,  3,  0, -5,-13,  6,  8, -6,  8,  3, -5,  1,  5, -1, 10},
{ 11,-17,  1, -2,  4,  5,  0,  3, 12, 12, -4,-10, -3,  0,  1, -2,  4, 15,  1,-13, 10, -3,  1,  2, -6,-11, -1, 11,  8,  8,  1,  5, -5, -3, -9, 13},
{ -1,  2, -9,  1, 14, -4,  3, -4,  3,  5,  3, -5, -4,  7,  2,  6, 17, 12,  5, -4, -9,  1, 22, -4, -8, -2, 15, -8,-11, -4,  6, 10,  5,  2, 14,  4},
{ -6,  2, -2,-13,  0,  0,  2, -3, -3, -1,  5, -3,-10,  3, -3, -9,-17, -9, -1, 11, 20,  0,  1, -5,  3,  7, -4,  4, 18, 10, -5, -4, -5, -2, -8, -4},
{ 17, 20, -7,-14,  3,  1, -7, -8, -4, 15, 22, -8,-14, -2, -2,-11, -6, -8, 14, 27,-12,  0,  5,  9,  3,  2, -1,  2, 19, -8,  4,  0, -2, -5,  0,-12},
{ 12, 11, -3, -4,  3,  5, -5,  2,  3,  4,  1, -8,  6,  9, -6, -9,  3,  4,  9, 14, -2,  4,  4, 10,  9,  9,  1,  1, 11, -4,  3,  1, -2,  1,  1, -5},
{  5,  1, -3, -1, 12,  2, -5,  9,  8,  0, -1,  3, -8, -4, -7, -5,  5,  4,  4, 19,  2,  4,  6,  1,  4, 11,  3, -2, 15,  5, -4,  1, -3,  3,  7,-13},
{  8, -2,  3,  5,  1, -5, -5,  2, 11,  0, -7,-11,  0, -3, -2,  5, 11, -2, 11, 14, 12, 11, 10,  7,  3, 13, 12,  9,  8, -3, 10, 16, 16,  0,  2, 12},
{  0, -8, -9,  0,  3, -3,-11, -3,  1,  1, -3, -6, -1, -2, -6,-17,-11, -5,  1,  4, -6,  8, 23, 13,  2, 13, 16, -6, 12,  6, 14, 11,  6,  0,  3, -5},
{  4, -2, -7, -3, -8, -7, -9, -4, -6, -3,  2,-12, -7, -6, -4, -9,  2,-12,  2,  5, -1, 10,  9,  6, -3, -2,-10,  8, 13, -8,  4,  3,  2,-11, -8,-17},
{-10,  1,-10,-13, -7,  2,-10, -2,-10,-13, -7,-13,  0, -8, -5,-16, -4, -8,  7, 10, -8,  1, 15, 16,  3,  0,  7, -6,  7,  0, 14, 13, 13, -6, -1, -9},
{  4, -3,-10,  0,-14, -3,-14, -1,  4, -6,  3,  1,-11,-19,-10,-17, -6,-13,  3,  4,  6, 15, 12, 14, -2,  6, -9,  0, 18, 10, 12,  7, 14, -1,  3,-15},
{-10, -4, -2, -9,-13, -5, -3, -2,-10,-14,-10,-13,-15,-15,-12, -9, -5, -8,  5, 10,  5, 16, 19, 28, 16, 24,  0,  0,  2,  4,  3,  2,  7,  4, 14, -2},
{ -2,  0, -2, -7, -7,  7, -5,  0,  0, -6, -5,  4, -7,-13, -9,-14, -3, 13,  0, -4,  3,  4, 13, 25, 13, 10,  7,-17,-18,  0, -4,  1,  7, -1, -2,  0},
{ -5, -6,  3, -2, -2,  2, -3,  3, 10,  4,-14,  3,  7, -3, -4, -1,-10,  4,  5,  2,  6, -2, -2,  3, -6,  5, 10,  6, -8, -1, 17, -2, -3, -5,-16,  0},
{ -6,-12,  1, 12, 15,  1,  1,-10, -5,  6, -3,  6,  7,  3, -6, -2, -4, 17, 11,-10,  1, -2,  1, -4, -3,-11,  9,  7,-16,  3,  9, 10,  6, -3,-13, 19},
{ -8,-10,  4,  0, -6,-15,-16,-20, -4, 10, 17, 15,  6,  2, -7, -2, -7,  3, -4,  4, 31, 29, 13, -9,-13,-26, -6, 35, 47, 44, 37,  7,  5, 10, -4, -1},
{  9,  2,-12,-12,  2, -9, -4, 15, 18, -8, -8,  1, 15,  1,-10,  9, 33, 12, 37, 17, -7, -3, -4, -8,  5, 24, 33,  4,  2, 24, 23,  0, -4, 25, 45, 32},
{ -4, 37, 55, 39,  3,-13, -1, 11,-10,  4, 32, 22,  8,-13,-21,-17,-11,-25,-17, 28, 50, 34, -2,  1, 17, 15, -6,-10,  9,  0, -8,-19,-15,-11,-10,-21},
{ 17, 33,  7,  9, -1, -1,  6,  9, -2,  8, 17, -8,  0, -4,  1,  5, 15,  2,  1, -1,-15,-14,-13, -6,  0,  4, -6, -9, -7,-17,-14,-25,-13,-10,  0,  0},
{ 20, 11,-10,-10, -8,  2,  0,  9,  5, 26, 28, -1,  2, -8,  4,  7, 16,  5,  5, -3,-13, -8,-14,-12,-13,  1,  1,  3, -2,-14,-10,-14, -8, -8,  2, -8},
{ 12, 29,  4,  9,-12,  1,  2, 11, -2, 14, 21, -3, -4, -5,  2, -1,  9,  4,  2,  4, -7, -1, -6, 10,  2,  8,-10,  5,  0,-11, -8,-10,  5, -9, -1,-12},
{ 23, 19,  3, -6, -6,  2,-13,  8, 20, 24, 15, -2, -3, -2, 13, -1, 15, 19, 10,  3, -7, -3,  2,  9, -8,  7, -3,-18, -3, -6, -6, -2, 10, -6,  2,-25},
{ 14,  5,-12,-10, -4, 14,  3, 16, 28,  4,  1,-13, -6,  0, 11,  3, 12, 22,-12,  0,-15,-16,-20,  9, -7,  5,  7, -5,  5,-20,-11, -3,  5,-19,  2, 13},
{  1,  2, -9, -3, -8, -6,-10,  1,  4,  9, 11, -8, -9, -9, -8,-18, -3,  8,  1, 12,-10, -6,-13, -9,-18,  4, 16,  2, 17, -5,-15,-19,-11,-16,  6,  7},
{ -1,  4,-16,-10, -7, -2,-10, -7, -2,  4,  2,-12,  0,-13, -3,-10, -5,  0,  8, 25, -6, -9, -6, -1,-11,  5, -5,  6, 11,-16, -8, -5,  1,-13,  3,  1},
{ -5, -1, -7,  0,-16, -8,-16, -8, -9,  3,  9,-10,  0,-12,-10, -9,  1, -2,  0, -1,-11, -6,-14, -7,-18, 12, 19,  4, 23,  0,  7,-13,-12,-11, 21, 25},
{  1, 15, -2,  4,-11, -3,  3,  7, -5, -9, -9, -4, -8,-22,-20, -6,  2,-16, 18, 27, 17, 27, 11, 10,  5, 16, 14, 14, 18, 15, 19,  2,  1,  5, 11, -2},
{-10,-11,  1, -3, 17, 14, 13,  6, -6,-19,-23,-13,-17, -8, -4, -9,-15, -9, 24, 23, 10, 13, 47, 36, 16, 20, 18,-14,-18,-12,-18,  7,  0,-14,-12, -7},
{  2,  6,  9, -4,  2, 11, -4,  1,  7, -6,-12, -3,  9,  2,  7, -4,-13, -8,  1, -1,  3, -2,  4, 10,  1,  6,  2, -3,-13, -3,  3,-11, 11, -5, -8, -8},
{ 18,-15, -5, -9,  0, -5,-13,-14, 22, 23,-21, -9,  2, 30, 10, -8, -9, 33, 22,-10, -8, -5,  1, -7, -9,-16, 16, 28, -7, -4,  7, 25,  5, -8,-16, 24},
{-12,-14, 16, 20, 25, 14,  4,-17, -7,-12, 30, 36, 34,  9, 21, 29, 10,-27,-10,-19,  9, 14, 23, 26, 17,-15,-14,-24,  8, 26, 16,  5, 37, 38, 36, -2},
{ 21, 25,  2,  8,  0,  2,  3, 15, 15,  5,  6, 12, 15,  0,  3, 13, 22, 16, 24, 20,  1, -2,  0,  9,  1, 23, 31, 23, 26,  8, -1, -3, 18, 18,  7,  8},
{-24, 13, 38, 34, 12, 20, 25, 15,-16,-27, -3, -8, -2, -5,  6,  6, -3,-26, -8, 18, 29, 19, 11, 35, 41, 13,-24,-22, -1,  5,  5,-10, -5, -6, -7,-28},
{ -8,  4, -6, -7,-11,  4,  4,  1,-14,-14, -3, -7,-11,-24, -7, -6, -7, -8,-14,  6,  5,  2,-16, -6, -1,  4, -8, -4, -3, -9,-11,-30,-12, -7, -6,-15},
{ -5, -8,  2,  3, -5, -2, -3, -2,  5, -8,  1, -1, -3,-14,  4,  3,  5,-14,  8,  2, -4, -2, -8,  2, -1, -9, -4,-12,  4, -6, -4,-17, -6,  1, -2,-17},
{-12, -4,  2,  9, -3, 17,  8, -1,-24,-15, -8,  2,  9, 14, 14,  8, -4,-21,-18, -2,  5, 12, -7,  7, -5, -8,-15,-18, -6,  2, 12, 11,  6, -7,-10,-22},
{ -3, -3,  3, 15, 11, 20,  6,  6,-13,-16, -6, -7,  7,  9,  7,-11,  4,-27,-12,  0,  5, 21,  9, 13, -9,  0,-16,-22,  3,-10, 16, 12,  6,-15,-10,-33},
{-10,  1, -4,  0,-14, -6,-10,  1,-13,  0, 26, -4,  1,  2, -2,-19,  5, -1, -9, -3,-17,  0,-15, -6, -2,  7,-21,  4,  7,-19,  2,  3, -6, -6, 28, -4},
{ -3, 23, -4,  0,  2,  0,-10,  6, -1, 11, 27,  2, -9, -8, -2, -6, 15,  5, -5,  7,-13, -3,  1, -4, -7, 22,  4,  3, 11, -4, -5, -7,-11,  4, 35,  8},
{ 10, 20, -5, -7, -1, -4,-11, 10, -9, 18, 10,-17, -7, -7, -4,-14,  5, -8,-12,  7, -6, -4, -2,-13, -2, 26, -2,-10,  8,-10, -2,-10,-12,-19, 21, 12},
{ 11,  4,-12,-12,-11, -7,-13,  1, -4, 12,  8, -5, -6,-19, -7, -3,  5, 16, -2, -5,-15, -6,-11,-11,-14, 14, 14, 25,  8, -4,-11,-22, -5, -1, 11,  9},
{  5,  6,  1,  9,  7,  8,  3,  3, 16,  7,  9,  5, 10,  3,  2,  8,  9,  4, 15,  6,  4,  0,  3, 15,  6,  8,  6, 10, 12,  6,  3, -2, 10,  8, 18,  4},
{  9, 11,  2, 12, 43, 26, 13, 10, 11,-16,-18, -8,-14,  4, -1, -9,-17,-16, 11, 12, 13, 30, 44, 13,  3, 16, 10,-19,-20, -8, -2,  4,-17,-15,-23,-21},
{  3, -4,  2,  3,  2, 10,  2,  0, -2,  8, -7, -4, -1, -9,  5, -4, -7,  3,  5,  0,  6, 12, -1,  2, -3, -6, -3,  8, -8,  1,  7,-11,  3, -4,-11,  9},
{  9,-10,  1, -1,  8, -1,  0, -6,  2, 19, -8, -3,  1, 11,  5, -1,-10,  8, -2,-12,  2,  6, 12,  7,  0, -9, -2, 17, -3, -1, -4,  6,  7,  2, -4, 11},
{ -6,-20,-12, -4, 17, 29, 31,  5, -7,  0, -5,  8,  5,  5, 32, 44, 42, 24,-11,-23,-20,-13, -4, -7,  0, -9,-10,  2,-10, -6, -1,  2,  5, 26, 20,  7},
{ 34, 19,  2, -6, -5, -2, -6, 15, 31, 34, 41, 13, -1, -2, 22, 26,  0, -2, 23, 12, -8, -8,  2,-13, -8,  0,  8, 13, 34,  1, -3,  1, 12,  5, -1,-12},
{ -3, 17, 11, -1, -4, 36, 41, 24,-18,-18, -9,-11,-11,-22,  0,  7, 17, -6,-10, 14, -6, -9,  5, 37, 53, 35, -9,-27,-20,-25,-19,-14,  9, 24, 34,  2},
{  1,  3, -1, -6,-16, -9,-14,  6,  7,  5,  5, -8, -9,-23,-11,-20, -8, -9, -1,  8,  0, -1, -3,  6,  5, 30, 19,  3, 15,  0, -1, -8, -4,-10, 14,  3},
{  4,  5,-16, -6,-13, -8,-18, -5,  4, -6,  2,-13, -7,-14,-12,-12, -2,  4,  2,  7, -1,  8, -4, -9, -7, 16, 19,  5, 18,  6,  6, -7,  2,  6, 29, 20},
{ -6,  4,  0,  7,-10, -4, -9, -1,  8, -8, -6, -9,  6,-17,-13,-13,  4,  6, -5, 12,  6,  5,-11, 12,  6, 27, 12,  4,  7,  6,  6, -3, -8, -3, 24, 13},
{  6,  2,-11,  8,-14, -6,-11, 10,  5,-15,  1,-13,  8,-10, -4,-13, -4,-23, 19,  1, -6,  3,-15, -5,  4, 29, 26, 21, 12,  2, 12, -5, -2, -9, 10, 24},
{ 14,  5,-14,  6,-24,-14,-11,  3,-20, 15, -1,-20,  3, -7,-14,-18,  8,-13, 26, 12, -2, 12, -5,-12,-12,  7, 11, 17, 11,  0,  3,  1, -8,-17,  5,  7},
{ 14, -4,-16, -7,-10, -6,-13, 10, 10, 10, -6,-16,-10,-14,-12, -7, 13, 10,  2, -1, -8, -5, -6, -2,-14,  6,  4,  5, -7,-12, -5, -4, -9,-14, 14,  8},
{ -2, -1,-10,  0, -3,-13, -3, 21,  4,  3,  2,-14, -3, -6, -7, -8, 13,  6,-10, -8,-12, -2, -6,-14,-15,  4, -5, -4, -8,-17, -9,-11, -3, -9,  6,  2},
{ 23,  6,-16, -4,-14, -4, -5,  3,  3, 25, 18, -8,-10,-13,  0,  2, 17,  6, -7,-14,-17, -8, -8, -1, -2,  6,-14, -7, -8,-11, -5, -7,-11, -7, 11, -4},
{ 18, 17, 10, 10, 16, 24, 19, 27, 18,  2, 10,  8,  2,  6, 22, 13, 21,  7, -4, 11, 10,  5,  3,  1,  4, 20,  2,-10,  2,  1,-19,-15,-10, -1, -4,-13},
{ 18, 18, 17, 35, 44, 10,  9, 25, 21, -5,-16,-13,  4, 14,-19,-11, -9, -5,  1,  2, 13, 16, 19,  2,  8, -2,-11,-11,-15,-11, -4, -3,-17,-14,-18,-19},
{  5,  0, -5,  8,  2, -1, -1, -2,  0, -6,-10,  1, 11, -8,  5, -3,-15, -5,  8,  2, -1, 10,  0,  4,  7,  6,  6, -6,-14,  3, 15,  0, 13, -4,-14, -6},
{  3, -5, -2,  3, -1, -4, -8,-11,  9,  9,  2,  4,  2, -5,  0,  2, -4, 14,  0,-10, -9,  4, 16,  3, -7, -2, 17,  7,  3,  5, 16, 12, 12, 10,  1, 22},
{ -1,-10, -9,  6,  1,-10,  0, -4,  4, 10, -9, -1, -1, -2, -5,  7,  4,  4,  6, -1, -2, 10,  5, -2,  0, -8, 12, 16, -1,  1,  7, -1, -2, -6, -6, 13},
{ 10, -6, -5, -2, 17, -2, -8, -5,  2,  3, 10,  0,  9, 11, -1, -8,-11,-10,  2,-11,  1, -1, 13,  3, -9,  3,  1,  7, 10,  3,  3,  5,  0, -1,  8,  3},
{  3,  8, -1, -2,  5, -2,  8,  5, -2, -6, -6, -3,  3,  0,  2,  6, 16,  5, -3,  0,  3, -3,  0,-12, -5, -1, -9, -6,-13, -5,  3,  1, -9,  0,  2,  4},
{  2,  5,  3,  7,  3, -6,-16, 23, 16,-10,  1, -7, -5, -1, -2,-12, 10, -3, -3, -8, -2,  1,  1,-12, -6, 17, 20, -2, -4, -7, -3, -3,-13,-16, 19, 16},
{  1,  4,  7, 12,  8,  3, -4, 12,  3, -3, -4, -2, -1, -3,  7,  3,  8, -2,  1,  2,  0, 10,  2, -4, -8, 12, 19,  1,  4,-12, -5,  3,  4, -9,  4,  0},
{  4,  7,  4,  0,  4,  9,  3, 10,  4,-12,  6,  8, -1,  1, -2,  3, 14,  1,  8,  8, -4,  2, 11,  1, -3,  0,  3,  0,  6, -5, -6, -6, -2,  7, -1, -4},
{  5,  5,  5,  4,  4,  5,  1, 12, 13,  3, -1,  1, 13, 15, 13, -3,  5,  8,  6,  6, -5, -4, -2,  5,  2, -3,  4,  2, 12,  5, -2,  3,  3, -4, -7, -2},
{  7,  6,  3, 17, 25, 10, -1,  5,  4,  0, -2,  2, 11, 17, 14,  5, 12, -3, -4, -6, -8,  1,  8, -3, -7, -6, -3, -4,-12, -8, -2,  1, -1, -8,  1, -2},
{ -5, -7,  0, 10, 14, 13, -4,  7,  6,-13,-12, -4,  7,  5,  1, -6, 20,  6, -4, -6, -4, -3, -6,  6,-12, -6, -2, -8,  0, -7, -1, -8, -4, -9,  0,-10},
{  7, -3,  5, 18, 20,  2, -2, 13,  7, -2, -4, -6, 12, 14,  8, -2,  2, -4, -5,  0, -7,  2, -2, -3, -4,  3, -8, -4, -4,-13, -9, -7,  4,-14, -9,-18},
{  1,  0, -1, 16, 12,  9,  4, 10,  5, -6,  2,  3, 17,  6,  5, 11, 16,  2,  6, -1,-12, -3,-10,  8, -4, -1, -2,-15, -6,-15, -9,-11, -8,  2,  4,-14},
{  4, 18, 21, 23, 14, 13, 11, 10,  9,  3,  8, 13,  7,  4,  3, 10,  1,  1, -3, -1, -3,-10,-11, -3, -1,  0,  0, -9, -9, -6,-14,-10,-11,-12,-14,-13},
{ 10,  4, 15, 24,  4,  3,  7,  0,  0,  4, -4,  2,  2, -2, -6, -7,-19,-19,  6,  0,  2,  7, -6, -6, -1,  0, -3, 15, -6,-11,-10,-14, -4,  2, -6, -5},
{ 12,  2, -2,  1, -4,  2,  0, -2,  0,  6,-19, -2, -3, -3, 10, -3, -9,  4, 15, -3, -2,  8,-11, 10,  6, -6, -2,  9,-15,  7,  6, -8, 11, -2,-19, 10},
{  6, -7, -9, -5, 20, 10, -8, -4, 14,  5, -6,  6,  2,  9,  9,  7,  4, 12, 13, -6,  2,  0, 19, 13, -6,-16, 12, 14, -8, 12, 11,  9,  3,  1, -8, 16},
{  0,  0,  5,  3,  5,  1, -2, -2,  3,  3,  0,  2,  9,  7,  2, -2,  2,  7,  8, -6,  0,  0,  9, -6,  1, -8,  5, 16, 11, 10,  5,  9, 12, 16, -3, 11},
{ -5, -3,  2,  4,  1, -2, -5, -2, -5, -3, 11,  5,  3, -1, -1,  4,  3, -2,  6,  9, 11,  9, 13,  7,  8, -5,  2,  9,  7,  6, 15,  7, -2,  8,  4, 11},
{ -3,  5, -1,  1, -2,  0,  6, -2, -6,  4,  1,  1,  3, -3, -5,  0, -8, -3,  8,  1,  2,  9, -1, -6,  3,  8,  6, -4,  5, -1,  1,  2, -8, -9, 10, -4},
{  4,  8,  9,  7, 11, -1, -3, -6, -2, -2,  7,  0,  1,  8, -8,-11,-12, -5,  1,  6,  2,  1, 19, -8, -4, 15, -6,  1,  3, -6, -6,  4,-12, -6, -6, -5},
{  2,  8,  5, 11, 18,  6, -8, -7,  2, -4,  4, -9, -6,  2, 10, -9,  3,  3,  3, -1,  1,  1, 19,  3,-12,-11,-10,  3, -2, -4,-10,-11,  3,-15, -5, -6},
{ -4,  9,  1,  3, 16, 10, -5,  1,  6, -4,  4, -7,  1, -1,  4, -2, -1,-10,  1,  3, -2, -2,  6,  7, -5,  2,  0, -5,  5,-19, -6,  2, -1, -2,  1,-10},
{ 11, 13, -2,  8,  6, 10,  4,  2, 12,  0, 14,  6,  2, -3, -1,  1,  2,  8, -4,  6,-13,  2,  2, -2, -3,  6, -8, -2, 11,-10,  3, -6, -4, -5, -2, -6},
{  5,  6,  6,  5, -6, -3,  1,  5,  7,  5,  0,  4, -4, -7,  6, -1,  5,  0,  0,  4, -2,  4,  3,  2,  4, -5, -1, -1,  0,  0, -6,  0,  8, -3,  0, -6},
{  2,  2,  7,  1,  0, 15, -5,  0,  4, -1,  4,  4,  5, -4, 12, -3,  8, -8,  2,  1, -9,  1,  4,  3,-17, -5, -3,  1,  2,-10,  0, -8, -4,-17, -9, -1},
{ -1,  5,  4,  6, -4, 12,  2,  8, -6, -2, -1, -3, -2,  1, 16,  4,  5,-10, -1,  0, -8, -1, -1, 10, -9, -7, -7, -1, -2, -6, -1,  1,  0, -7,-10, -4},
{  5,  4,  3,  3,  1, 14, -1, -6, -5,-12, -1, -3,  2,  1, -2, -7, -5,-10,  1,  8, -5,  2, -2, -1, -4, -5, -2, -6,  4,-11, -4, -7, -2, -9,-11, -6},
{  5,  4, 10,  1,  8,  8,  4,  3,  9, -4, -1, 11, -3,  7, -2,  3, -3,-14,  4,  0, -3, -5,  6,  5,  7, -2,  7, -4,  2, -8, -7, -5,-12, -4, -7,-13},
{  9,  1,  4,  0,  7, -2,  3,  8,-10, 11,-10, -8, -3,  8,  5,  6, -1, -5,  7,  8, -1,-10, -5,  2, 12, 13, -5,  2,-10, -5, -6, 12,  0,  4,  3,  0},
{  4,-11, -6, 10, -7,  2,  4, -8, -4,  4, -9,  1,  7,-10, -3, -7,-17,  4,  9, -8, -7,  8,  4, -2, -5, -2,  0, 18, -1, -4,  0,  0,  4, -3, -5, 14}
};

#define SVM_base ((-427)<<AMP_Svm)  /*!< The integer offset used in the svm algorithm. */

/*!
 * @brief Applying the SVM algorithm to detect the pedestrian with the standard sample vector detector[].
 * @details The linear SVM algorithm is applied to find out the match to pedestrians.
 * At high level, this evaluates a vector dot product comparing against a vector obtained through training over a set of sample images.
 * If the result is above a threshold, it indicates a match.
 * @note This matches the gpu::HOGDescriptor::detect() in the OpenCV library.
 * @param[out] results : The output to the host.
 * @param[in] blX : The actual number of blocks per image row.
 * @param[in] blY : The actual number of blocks per image column.
 */
void kernel svm(global int *restrict results, int blX, int blY) {
   // The dot product needs to accumulate data across a block that spans
   // multiple rows; use a rotating register to accumulate the partial sums
   //
   // The original OpenCV implementation uses a cache to store the inputs, as
   // they are reused across multiple dot products; instead, this implementation 
   // determines the contribution of the current input to all the dot products
   // and keeps multiple dot-products in flight
   //
   // Use a second register to extract the computed data from the rotating
   // register and output it in natural order
   //
   // every MAXBX / WX * BLOCK_HIST decimate one element
   // every MAXBX / BX * BLOCK_HIST shift rows up
   short sums[WY][MAXBX_SVM + WX + 1 + 2];
   short sumbuf[WY][WX];
   // all sums are initially 0
   #pragma unroll
   for (int y = 0; y < WY; y++) {
      #pragma unroll
      for (int x = 0; x < MAXBX_SVM + 3+ WX; x++) {
         sums[y][x] = SVM_base;
      }
      #pragma unroll
      for (int x = 0; x < WX; x++) {
         sumbuf[y][x] = SVM_base;
      }
   }

   int count = 0;
   int col = 0, row = 0;
   uchar hist = 0;
   bool newRow=false, newCol=false;

   // Apply the current result to all the dot products currently in flight

   for (int i = 0; i < (MAXBX_SVM+WX-1) * (blY + 2) * BLOCK_HIST; i++) {
      short data=((row < blY) && (col < blX))?(short)read_channel_altera(chNorm_Svm):0;
      newCol=false;
      /* Compute the inner product for the read data simultaneously. */
      #pragma unroll WY
      for (int y = 0; y < WY; y++) {
         #pragma unroll
         for (int x = 0; x < WX; x++) {
            short tmp1=( (data&0xffff) * (detector[x *  WY + (WY - 1 - y)][hist]&0xffff) );     
            bool fu= (((sumbuf[y][x]>>14)&0x3)==0x2);
            sumbuf[y][x]+= ((tmp1)>>1);
            if (fu && sumbuf[y][x]>0 ) {sumbuf[y][x]=-32768; }
         }
      }

      // Keep track of the current image position
      // this hops in line with the rotating register
      hist++;
      if (hist >= BLOCK_HIST) {
         hist=0;
         col++;
         newCol = true;
         if (col >= MAXBX_SVM+WX-1) {
            col = 0;
            row++;
         }
      }

      // rotate the register by WX
      #define Svm_SRcols 3 /*!< Shifting count per cycle */
      #pragma unroll
      for (int y=0; y<WY; ++y)  {
    	  short tmpsum[Svm_SRcols];
          #pragma unroll
    	  for (int x=0;x<Svm_SRcols; ++x)  tmpsum[x]=sums[y][x];
          #pragma unroll
          for (int x=0; x<MAXBX_SVM+WX+3-Svm_SRcols; ++x)  sums[y][x]=sums[y][x+Svm_SRcols];
          #pragma unroll
          for (int x=0;x<Svm_SRcols; ++x)  sums[y][MAXBX_SVM+WX+3-Svm_SRcols+x]=tmpsum[x];

      }
      // Figure out where to save the data in global memory
      if (hist == 0 && (row >= WY + 1) && (row <= blY + 1) && (col >= 0) && (col < blX-WX+1)) {
         results[(row - WY - 1) * (blX - WX + 1) + col ] = sums[0][MAXBX_SVM+WX-1];
      }

      // Logic required to extract the data from the rotating registers
      // This is needed to ensure that the compiler can implement this
      // efficiently in hardware
      // After unrolling all transfers are between pairs of statically known locations in the
      // array
      if (newCol) {
          #pragma unroll WY
          for (int y = 0; y < WY; y++) {
             sums[y][WX]=sumbuf[y][WX-1];
             #pragma unroll 
             for (int x = WX - 2; x >= 0; x--) {
                sumbuf[y][x+1]=sumbuf[y][x];
             }
          }
          #pragma unroll
          for (int y=0; y<WY-1; ++y)  {
              sums[y][0]=sums[y+1][MAXBX_SVM+WX-1];
              sumbuf[y][0]=sums[y][0];
          }
          sumbuf[WY-1][0]=sums[WY-1][0]=SVM_base;
      }

   }
}

#define SCALE_GRAN 256 /*!< Represent scale as fraction SCALE_GRAN / ratio */

/*!
 * @brief Input an image of ROWS x COLS, and output is a reduced scale image by a factor SCALE_GRAN / ratio
 * @note It is the equivalent of gpu::HOGInvoker::resize() of OpenCV library.
 * @param[in] ratio : Describes the scaling factor
 * @param[in] image_input : The original input image.
 * @param[in] rows : The number of rows in the original image.
 * @param[in] cols : The number of columns in the original image
 */
void kernel resize(unsigned ratio, global uchar4 * restrict image_input, int rows, int cols) {
   unsigned int out_pointer = 0;
   uchar4 buf[MAXCOLS + 2]; // Must buffer one row and 2 pixels from the original image to be able to downscale using bilinear interpolation
   uint accumColumn, accumRow = SCALE_GRAN - ratio;
   int col = 0, row = 0;
   bool write_x, write_y;
   int linear_in = 0;

   // Downsample the image using vector operation. It processes row by row.
   for (int i = 0; i < rows * MAXCOLS + MAXCOLS + 1; i++) {
      if (row < rows && col < cols) {
         buf[MAXCOLS + 1] = image_input[linear_in];
         linear_in++;
      }

      // Use integer logic to compute the fraction and determine when to
      // write and how to interpolate
      // write this row / column only if overflowing ratio + SCALE_GRAN
      if (col == 0) {
         accumColumn = SCALE_GRAN - ratio;
         write_y = false;
         accumRow += ratio;
         if (accumRow >= SCALE_GRAN) {
            accumRow -= SCALE_GRAN;
            write_y = true;
         }
      }
      accumColumn += ratio;
      write_x = false;
      if (accumColumn >= SCALE_GRAN) {
         accumColumn -= SCALE_GRAN;
         write_x = true;
      }
      /* The actual bilinear interpolation core. */
      if (write_x && write_y && (i >= MAXCOLS + 1) && col > 0 && (col - 1) < cols) {
         uint4 pix0 = convert_uint4(buf[0]);
         uint4 pix1 = convert_uint4(buf[1]);
         uint4 pix2 = convert_uint4(buf[MAXCOLS]);
         uint4 pix3 = convert_uint4(buf[MAXCOLS + 1]);
         /* The following operations process all components in a pixel together. */
         uint rc=(accumRow&0xff)*(accumColumn&0xff);
         uchar4 pixel=convert_uchar4((pix3*SCALE_GRAN*SCALE_GRAN + 
                       ((pix1-pix3)&0x3ffffff)*(accumRow&0x3ffffff)*SCALE_GRAN + 
                       ((pix2-pix3)&0x3ffffff)*(accumColumn&0x3ffffff)*SCALE_GRAN + 
                       ((pix0+pix3-pix2-pix1)&0x3ffffff)*(rc&0x3ffffff))/SCALE_GRAN/SCALE_GRAN);
         write_channel_altera(chResize_Gradient, pixel);
      } 
      /* Shift the buffer in preparation for the next iteration. */
      #pragma unroll
      for (int i = 0; i < MAXCOLS + 1; i++) {
         buf[i] = buf[i + 1];
      }
      col = col == MAXCOLS - 1 ? 0 : col + 1;
      row = col == 0 ? row + 1 : row;
   }  
}

/*!
 * @details Compute the 2 weights for 2 bins per pixel.
 * @retval The absolute value of weight and the radius angle.
 */
float2 cartToPolar(short x, short y, uint xy) {
   float mag = sqrt((float)(xy));
   float a = atan2((float)y, (float)x);
   if (a < 0) a += (float)(2 * M_PI);
   return (float2)(a, mag);
}

/** @brief The look-up table to compute {x^2|0<=x<256}. */
constant unsigned short sq[256]={
0,  1,     4,     9,    16,    25,    36,    49,    64,    81,   100,   121,   144,   169,   196,  225, 256, 
  289,   324,   361,   400,   441,   484,   529,   576,   625,   676,   729,   784,   841,   900,   961, 1024, 
 1089,  1156,  1225,  1296,  1369,  1444,  1521,  1600,  1681,  1764,  1849,  1936,  2025,  2116,  2209, 2304, 
 2401,  2500,  2601,  2704,  2809,  2916,  3025,  3136,  3249,  3364,  3481,  3600,  3721,  3844,  3969, 4096, 
 4225,  4356,  4489,  4624,  4761,  4900,  5041,  5184,  5329,  5476,  5625,  5776,  5929,  6084,  6241, 6400, 
 6561,  6724,  6889,  7056,  7225,  7396,  7569,  7744,  7921,  8100,  8281,  8464,  8649,  8836,  9025, 9216, 
 9409,  9604,  9801, 10000, 10201, 10404, 10609, 10816, 11025, 11236, 11449, 11664, 11881, 12100, 12321, 12544, 
12769, 12996, 13225, 13456, 13689, 13924, 14161, 14400, 14641, 14884, 15129, 15376, 15625, 15876, 16129, 16384, 
16641, 16900, 17161, 17424, 17689, 17956, 18225, 18496, 18769, 19044, 19321, 19600, 19881, 20164, 20449, 20736, 
21025, 21316, 21609, 21904, 22201, 22500, 22801, 23104, 23409, 23716, 24025, 24336, 24649, 24964, 25281, 25600, 
25921, 26244, 26569, 26896, 27225, 27556, 27889, 28224, 28561, 28900, 29241, 29584, 29929, 30276, 30625, 30976, 
31329, 31684, 32041, 32400, 32761, 33124, 33489, 33856, 34225, 34596, 34969, 35344, 35721, 36100, 36481, 36864, 
37249, 37636, 38025, 38416, 38809, 39204, 39601, 40000, 40401, 40804, 41209, 41616, 42025, 42436, 42849, 43264, 
43681, 44100, 44521, 44944, 45369, 45796, 46225, 46656, 47089, 47524, 47961, 48400, 48841, 49284, 49729, 50176, 
50625, 51076, 51529, 51984, 52441, 52900, 53361, 53824, 54289, 54756, 55225, 55696, 56169, 56644, 57121, 57600, 
58081, 58564, 59049, 59536, 60025, 60516, 61009, 61504, 62001, 62500, 63001, 63504, 64009, 64516, 65025};

/*! @brief Takes resized image and pads it to max size then computes gradients.
 * @details Because the local object appearance and shape can offten be characterized rather well
 * by the distribution of local intensity gradients, the gradient() kernel computes the gradient
 * for each pixel as the input for histograms().
 * @note It is included in the gpu::HOGDescriptor::computeGradient() of OpenCV library.
 * @param[in] rows : The resized rows.
 * @param[in] cols : The resized columes.
 */
void kernel gradient(unsigned rows, unsigned cols) {
   unsigned col = 0, row = 0;
   uchar4 buf[2 * MAXCOLS + 1]; /*!< The shift buffer for original row of image*/
   /* It processes and computes the gradient pixel by pixel*/
   for (int i = 0; i < rows * MAXCOLS + MAXCOLS; i++) {
      if (col < cols && row < rows) {
         buf[2 * MAXCOLS] = read_channel_altera(chResize_Gradient);
      } else {
         buf[2 * MAXCOLS] = 0;
      }

      // Fetches the pixels required to compute the gradient
      uchar4 left = col == 0 ? buf[MAXCOLS] : buf[MAXCOLS - 1];
      uchar4 right = col == MAXCOLS - 1 ? buf[MAXCOLS] : buf[MAXCOLS + 1];
      uchar4 top = row == 1 ? buf[MAXCOLS] : buf[0];
      uchar4 bottom = row == cols ? buf[MAXCOLS] : buf[2 * MAXCOLS]; 
      uint mag;
      short dx, dy;
      /* Compute the current pixel gradient using the adjacent pixels data*/
      #pragma unroll 3
      for (int i = 0; i < 3; i++) {
         short dx0 = right[i] - left[i];
         short dy0 = bottom[i] - top[i];
         /* Use a look-up table to compute the square, which avoids using DSP */
         ushort xx0 = sq[dx0>=0?dx0:-dx0];
         ushort yy0 = sq[dy0>=0?dy0:-dy0];
         uint   mag0 = xx0+yy0;
         if ((i == 0) || (mag0 > mag)) {
            dx = dx0;
            dy = dy0;
            mag = mag0;
         }
      }
      float2 polar = cartToPolar(dx, dy, mag);
      polar.x = polar.x * (float)(NBINS / M_PI) - 0.5f; // fractional bin

      /* Because each gradient impacts two bins, and each bin is impacted with a scale of weight,
       * the following codes computes the impacted bin number and the related weight. */
      uchar2 bins;
      ushort2 weights;
      char hidx = floor(polar.x);
      float angle = polar.x - hidx;
      if (hidx < 0) hidx += NBINS; else if (hidx >= NBINS) hidx -= NBINS;
      bins.x = hidx;
      hidx++;
      hidx &= hidx < NBINS ? -1 : 0;
      bins.y = hidx;
      float yweight=polar.y * angle ;
      /* The aplifier is used to increae the precision for casting from floating point to short.*/
      weights.x = (ushort)(ldexp( polar.y - yweight ,AMP_Hist));
      weights.y = (ushort)(ldexp(           yweight ,AMP_Hist));
      if (row >= 1 && col < cols && row < rows + 1) {
         write_channel_altera(chGrad_Hist_weight, weights);
         write_channel_altera(chGrad_Hist_angle, bins);
      } 
      /* Shift the buffer in preparation for the next iteration. */
      #pragma unroll
      for (int i = 0; i < 2 * MAXCOLS; i++) {
         buf[i] = buf[i + 1];
      }
      col = col == MAXCOLS - 1 ? 0 : col + 1;
      row = col == 0 ? row + 1 : row;
   }
}

/*! @brief The constant weight for computing histograms. */
constant ushort weights[CELL_SIZE * CELL_SIZE * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE ] = {  // Multiple 1000
  5,  0,  0,  0, 11,  0,  0,  0, 20,  0,  0,  0, 32,  0,  0,  0,
 43,  0,  2,  0, 46,  0, 10,  0, 46,  0, 20,  0, 41,  0, 32,  0,
 33,  0, 42,  0, 23,  0, 50,  0, 12,  0, 54,  0,  3,  0, 53,  0,
  0,  0, 43,  0,  0,  0, 28,  0,  0,  0, 16,  0,  0,  0,  9,  0,
 11,  0,  0,  0, 22,  0,  0,  0, 39,  0,  0,  0, 63,  0,  0,  0,
 84,  0,  5,  0, 91,  0, 21,  0, 90,  0, 41,  0, 81,  0, 63,  0,
 65,  0, 83,  0, 45,  0, 99,  0, 24,  0,106,  0,  7,  0,105,  0,
  0,  0, 84,  0,  0,  0, 55,  0,  0,  0, 33,  0,  0,  0, 18,  0,
 20,  0,  0,  0, 39,  0,  0,  0, 69,  0,  0,  0,113,  0,  0,  0,
149,  0,  9,  0,161,  0, 37,  0,160,  0, 72,  0,143,  0,111,  0,
115,  0,148,  0, 79,  0,175,  0, 43,  0,189,  0, 12,  0,186,  0,
  0,  0,149,  0,  0,  0, 98,  0,  0,  0, 58,  0,  0,  0, 32,  0,
 32,  0,  0,  0, 63,  0,  0,  0,113,  0,  0,  0,184,  0,  0,  0,
244,  0, 16,  0,263,  0, 60,  0,260,  0,118,  0,234,  0,182,  0,
187,  0,241,  0,130,  0,286,  0, 71,  0,307,  0, 20,  0,303,  0,
  0,  0,244,  0,  0,  0,159,  0,  0,  0, 95,  0,  0,  0, 52,  0,
 43,  2,  0,  0, 84,  5,  0,  0,149,  9,  0,  0,244, 16,  0,  0,
323, 21, 21,  1,348, 23, 80,  5,344, 23,156, 10,310, 20,241, 16,
248, 16,319, 21,172, 11,378, 25, 94,  6,407, 27, 26,  1,402, 26,
  0,  0,323, 21,  0,  0,211, 14,  0,  0,126,  8,  0,  0, 69,  4,
 46, 10,  0,  0, 91, 21,  0,  0,161, 37,  0,  0,263, 60,  0,  0,
348, 80, 23,  5,376, 86, 86, 20,372, 85,169, 39,334, 77,260, 60,
268, 61,344, 79,185, 42,408, 94,101, 23,439,101, 28,  6,434,100,
  0,  0,348, 80,  0,  0,228, 52,  0,  0,136, 31,  0,  0, 74, 17,
 46, 20,  0,  0, 90, 41,  0,  0,160, 72,  0,  0,260,118,  0,  0,
344,156, 23, 10,372,169, 85, 39,368,167,167, 76,330,150,257,116,
265,120,341,155,183, 83,404,183,100, 45,435,197, 28, 13,429,195,
  0,  0,344,156,  0,  0,225,102,  0,  0,135, 61,  0,  0, 73, 33,
 41, 32,  0,  0, 81, 63,  0,  0,143,111,  0,  0,234,182,  0,  0,
310,241, 20, 16,334,260, 77, 60,330,257,150,116,297,231,231,179,
238,185,306,238,165,128,363,282, 90, 70,390,304, 25, 20,385,300,
  0,  0,310,241,  0,  0,202,157,  0,  0,121, 94,  0,  0, 66, 51,
 33, 42,  0,  0, 65, 83,  0,  0,115,148,  0,  0,187,241,  0,  0,
248,319, 16, 21,268,344, 61, 79,265,341,120,155,238,306,185,238,
191,246,246,316,132,170,291,374, 72, 93,313,403, 20, 26,309,398,
  0,  0,248,319,  0,  0,162,209,  0,  0, 97,125,  0,  0, 53, 68,
 23, 50,  0,  0, 45, 99,  0,  0, 79,175,  0,  0,130,286,  0,  0,
172,378, 11, 25,185,408, 42, 94,183,404, 83,183,165,363,128,282,
132,291,170,374, 91,201,201,444, 50,110,217,477, 14, 31,214,471,
  0,  0,172,378,  0,  0,112,247,  0,  0, 67,148,  0,  0, 36, 81,
 12, 54,  0,  0, 24,106,  0,  0, 43,189,  0,  0, 71,307,  0,  0,
 94,407,  6, 27,101,439, 23,101,100,435, 45,197, 90,390, 70,304,
 72,313, 93,403, 50,217,110,477, 27,118,118,514,  7, 33,117,507,
  0,  0, 94,407,  0,  0, 61,266,  0,  0, 36,160,  0,  0, 20, 87,
  3, 53,  0,  0,  7,105,  0,  0, 12,186,  0,  0, 20,303,  0,  0,
 26,402,  1, 26, 28,434,  6,100, 28,429, 13,195, 25,385, 20,300,
 20,309, 26,398, 14,214, 31,471,  7,117, 33,507,  2, 33, 33,500,
  0,  0, 26,402,  0,  0, 17,263,  0,  0, 10,157,  0,  0,  5, 86,
  0, 43,  0,  0,  0, 84,  0,  0,  0,149,  0,  0,  0,244,  0,  0,
  0,323,  0, 21,  0,348,  0, 80,  0,344,  0,156,  0,310,  0,241,
  0,248,  0,319,  0,172,  0,378,  0, 94,  0,407,  0, 26,  0,402,
  0,  0,  0,323,  0,  0,  0,211,  0,  0,  0,126,  0,  0,  0, 69,
  0, 28,  0,  0,  0, 55,  0,  0,  0, 98,  0,  0,  0,159,  0,  0,
  0,211,  0, 14,  0,228,  0, 52,  0,225,  0,102,  0,202,  0,157,
  0,162,  0,209,  0,112,  0,247,  0, 61,  0,266,  0, 17,  0,263,
  0,  0,  0,211,  0,  0,  0,138,  0,  0,  0, 83,  0,  0,  0, 45,
  0, 16,  0,  0,  0, 33,  0,  0,  0, 58,  0,  0,  0, 95,  0,  0,
  0,126,  0,  8,  0,136,  0, 31,  0,135,  0, 61,  0,121,  0, 94,
  0, 97,  0,125,  0, 67,  0,148,  0, 36,  0,160,  0, 10,  0,157,
  0,  0,  0,126,  0,  0,  0, 83,  0,  0,  0, 49,  0,  0,  0, 27,
  0,  9,  0,  0,  0, 18,  0,  0,  0, 32,  0,  0,  0, 52,  0,  0,
  0, 69,  0,  4,  0, 74,  0, 17,  0, 73,  0, 33,  0, 66,  0, 51,
  0, 53,  0, 68,  0, 36,  0, 81,  0, 20,  0, 87,  0,  5,  0, 86,
  0,  0,  0, 69,  0,  0,  0, 45,  0,  0,  0, 27,  0,  0,  0, 14
};

#define MAXBLOCKS (((MAXBX + 1) / 2) * 2) /*!< the number of blocks in the rotating registers */

/*! @brief Takes gradients and creates histograms.
 * details It divides the image windows into small spatial regions called cells.
 * For each cell, it accumulates a local 1-D histogram of gradient directions over the pixels of the cell.
 * The combined histogram entries from the representation.
 * It uses a rotating register to keep (block / cell)^2 sets of blocks in flight.
 * Each cell impacts the histogram of up to 4 cells in the block and is involved in the computation of 4 blocks,
 * including the top left, top right, bottom left and bottom right block.
 */
/* The position is illustrated as :
 *    _____________________________
 *    |   top left   | top right  |
 *    |   block      | block      |
 *    |              |            |
 *    |------(current pixel)------|
 *    |              |            |
 *    |  bottom left |bottom right|
 *    |  block       | block      |
 *    |______________|____________|
 */
/*! After finishing processing a row of cells, the top buffer is ready to be written,
 * and the bottom buffer is shifted into top buffer served as the top blocks for the next row of cells.
 *
 * The detection algorithm runs in 4 nested loops (at each pyramid layer):
   - loop over the windows within the input image
   - loop over the blocks within each window
   - loop over the cells within each block
   - loop over the pixels in each cell
 * In the histograms() kernel, two long buffers are created to unroll the loop over the cells and pixels.
 * @note It is included in the gpu::HOGCache::getBlock() of OpenCV library.
 * @param[in] rows : The resized rows.
 * @param[in] cols : The resized columes.
 * @param[in] delta : The margin colume value of the image.
 */
void kernel histograms(int rows, int cols, int delta) {
   /* The two buffers to unroll the computation over pixels and cells. 
    * The top_histograms stores the upper cell row, The bottom_histograms store the lower cell row.
    * The out_histograms stores the block ready to transmit to next kernel normalizeit(). */
   ushort top_histograms[(MAXBLOCKS + 3) * BLOCK_HIST];
   ushort bottom_histograms[(MAXBLOCKS + 3) * BLOCK_HIST];
   ushort out_histograms[BLOCK_HIST];

   int row_in_cell = 0; // varies from 0 to 7 for each row of cells
   int offset = 0;      // varies from 0 to 15 
   int base = 0;        // increments by 16, up to MAXCOLS
   int block_count = 0; // base / 16
   int row = 0;
   unsigned int wBlk=0, wCellBin=BLOCK_HIST;
   #pragma unroll
   for (int i = 0; i < (MAXBLOCKS + 3) * BLOCK_HIST; i++) {
      top_histograms[i] = bottom_histograms[i] = 0;
   }

   #pragma unroll
   for (int i = 0; i < BLOCK_HIST; i++) {
      out_histograms[i] = 0;
   }

   /* It processes and computes the histogram pixel by pixel.
    * First it fetches the weights and bins of each pixel.
    * Then the weights are accumulated into the histograms in the 4 adjacent cell.
    * Finally the buffers shift for the next block.
    * Because the histograms() send a whole row of pixels after another,
    * each cell is scattered across 8 segments.
    * The same buffer will be used to accumulated these segments. */
   for (int i = 0; row < (((unsigned)rows) / CELL_SIZE + 2) * CELL_SIZE; i++) {
      int col = base + offset;
      uchar2 bins = (uchar2)(0,0);
      ushort2 w = (ushort2)(0, 0);
      if (col >= delta && col < cols - delta && row >= delta && row < rows - delta) {
         w = read_channel_altera(chGrad_Hist_weight);
         bins = read_channel_altera(chGrad_Hist_angle);
      }

      uchar bin0 = bins.x, bin1 = bins.y;
      ushort weight0 = w.x, weight1 = w.y;

      /* This data is involved in the computation of 4 blocks.
       * For each block, the data is involved in the computation of 4 histograms.
       * It reads the 4 blocks at 4 locations each, including 
       *    top_left    |   top_right
       *    bottom_left |   bottom_right */
      #pragma unroll 4
      for (int j = 0; j < BLOCK_SIZE * BLOCK_SIZE; j++) {
         ushort bin0_top_left = 0, bin1_top_left = 0;
         ushort bin0_top_right = 0, bin1_top_right = 0;
         ushort bin0_bottom_left = 0, bin1_bottom_left = 0;
         ushort bin0_bottom_right = 0, bin1_bottom_right = 0;
         /* Four temporary 1st-level cache buffers tms, tns, bms, tms are used for compute the adjacent 8 pixels in the same row of the cell based on 4 directions.
          * They first fetch the data from the 2nd-level caches over the long buffers according to the current position of the pixel.
          * Then the weights of the pixels are accumulated in the temporary variables.
          * Finally the caches are written back into the 2 long buffer */

         ushort tms[NBINS], tns[NBINS], bms[NBINS], bns[NBINS];
         #pragma unroll
         for (int k=0; k<NBINS; ++k)  {
             tms[k]=top_histograms   [BLOCK_HIST+j*NBINS+k];
             tns[k]=top_histograms   [j*NBINS+k];
             bms[k]=bottom_histograms[BLOCK_HIST+j*NBINS+k];
             bns[k]=bottom_histograms[j*NBINS+k];
         }
         bin0_top_left=tms[bin0]; bin0_top_right=tns[bin0];
         bin0_bottom_left=bms[bin0]; bin0_bottom_right=bns[bin0];
         bin1_top_left=tms[bin1]; bin1_top_right=tns[bin1]; 
         bin1_bottom_left=bms[bin1]; bin1_bottom_right=bns[bin1];

         unsigned index_bottom_right =  (row_in_cell              * CELL_SIZE * BLOCK_SIZE + (offset & 0x07)            ) * BLOCK_SIZE * BLOCK_SIZE + j;
         unsigned index_bottom_left =  (row_in_cell              * CELL_SIZE * BLOCK_SIZE + (offset & 0x07) + CELL_SIZE) * BLOCK_SIZE * BLOCK_SIZE + j;
         unsigned index_top_right =    ((row_in_cell + CELL_SIZE) * CELL_SIZE * BLOCK_SIZE + (offset & 0x07)            ) * BLOCK_SIZE * BLOCK_SIZE + j;
         unsigned index_top_left =    ((row_in_cell + CELL_SIZE) * CELL_SIZE * BLOCK_SIZE + (offset & 0x07) + CELL_SIZE) * BLOCK_SIZE * BLOCK_SIZE + j;

         /* Accumulated into the current 4 histograms */
         bin0_top_left    += ( (weights[index_top_left]   &0xffff) * (weight0&0xffff) ) >> SHR_Inhist;
         bin1_top_left    += ( (weights[index_top_left]   &0xffff) * (weight1&0xffff) ) >> SHR_Inhist;
         bin0_top_right    += ( (weights[index_top_right]   &0xffff) * (weight0&0xffff) ) >> SHR_Inhist;
         bin1_top_right    += ( (weights[index_top_right]   &0xffff) * (weight1&0xffff) ) >> SHR_Inhist;
         bin0_bottom_left += ( (weights[index_bottom_left]&0xffff) * (weight0&0xffff) ) >> SHR_Inhist;
         bin1_bottom_left += ( (weights[index_bottom_left]&0xffff) * (weight1&0xffff) ) >> SHR_Inhist;
         bin0_bottom_right += ( (weights[index_bottom_right]&0xffff) * (weight0&0xffff) ) >> SHR_Inhist;
         bin1_bottom_right += ( (weights[index_bottom_right]&0xffff) * (weight1&0xffff) ) >> SHR_Inhist;

         /* Write back the updated values. */
         #pragma unroll
         for (uchar k = 0; k < NBINS; k++) {
           if (k == bin0) {
              top_histograms   [BLOCK_HIST + j * NBINS + k] = bin0_top_left;
              top_histograms   [             j * NBINS + k] = bin0_top_right;
              bottom_histograms[BLOCK_HIST + j * NBINS + k] = bin0_bottom_left;
              bottom_histograms[             j * NBINS + k] = bin0_bottom_right;
           } else if (k == bin1) {
              top_histograms   [BLOCK_HIST + j * NBINS + (k)] = bin1_top_left;
              top_histograms   [             j * NBINS + (k)] = bin1_top_right;
              bottom_histograms[BLOCK_HIST + j * NBINS + (k)] = bin1_bottom_left;
              bottom_histograms[             j * NBINS + (k)] = bin1_bottom_right;
           }
         }
      }

      /* Shift the buffers to match the next cell's coordinates.
       * The 2 buffers will always put the current cells to be accumulated into the 3rd (left) and the 4th (right) elements. 
       * Hence the data for the current blocks can be always fetched from the fixed position instead of index-based position. */
      #define Hist_Range (MAXBX*BLOCK_HIST) /*!< The index range of shift register */
      #define Hist_SR ((MAXBX+1)*BLOCK_HIST/CELL_SIZE) /*!< Right shift for buffer per cycle */
      #define Hist_SRcols (MAXCOLS/2) /*!< Shifting count per cycle */
      #pragma unroll
      for (int i=0;i<Hist_Range/Hist_SRcols;++i)  {
          ushort t=   top_histograms[(Hist_SR*(Hist_SRcols-1)+i)%Hist_Range+BLOCK_HIST*2];
          ushort b=bottom_histograms[(Hist_SR*(Hist_SRcols-1)+i)%Hist_Range+BLOCK_HIST*2];
          #pragma unroll
          for (int j = Hist_SRcols-1; j >0 ; --j) {
              top_histograms   [(Hist_SR*(j)+i)%Hist_Range+BLOCK_HIST*2] = top_histograms   [(Hist_Range+Hist_SR*(j-1)+i)%Hist_Range+BLOCK_HIST*2];
              bottom_histograms[(Hist_SR*(j)+i)%Hist_Range+BLOCK_HIST*2] = bottom_histograms[(Hist_Range+Hist_SR*(j-1)+i)%Hist_Range+BLOCK_HIST*2];
          }
          top_histograms   [(Hist_SR*(0)+i)%Hist_Range+BLOCK_HIST*2]=t;
          bottom_histograms[(Hist_SR*(0)+i)%Hist_Range+BLOCK_HIST*2]=b;
      }

      /* Implement the transfers between the top and bottom row buffers
       * It occurs when the 8 adjacent pixels are completed and before turning to the next cell. */
      if ( (offset&0x7) ==7 )  {
         #pragma unroll
         for (int i = 0; i < BLOCK_HIST; i++) {
             top_histograms   [i+BLOCK_HIST*2]=top_histograms   [i+BLOCK_HIST];
             top_histograms   [i+BLOCK_HIST]  =top_histograms   [i];
             top_histograms   [i]             =top_histograms   [i+BLOCK_HIST*MAXBLOCKS];
             bottom_histograms[i+BLOCK_HIST*2]=bottom_histograms[i+BLOCK_HIST]; 
             bottom_histograms[i+BLOCK_HIST]  =bottom_histograms[i]; 
             bottom_histograms[i]             =bottom_histograms[i+BLOCK_HIST*MAXBLOCKS];
         }
      }

      /* When the last row of the cell (i.e. 8th row) is completed, the data can be sent to the next kernel.
       * The ready block is temporarily stored in the out_histograms buffer, which is then sent onto channels. 
       * Then the bottom block is shifted into the top block. */
      bool shift_cell_rows = (row >= CELL_SIZE)&&(row_in_cell == 7) && ((offset&0x7) == 7);
      if (shift_cell_rows) {
         #pragma unroll
         for (int j = 0; j <  BLOCK_HIST; j++) {
            out_histograms[j]=top_histograms[j+BLOCK_HIST*2];
            top_histograms[j+BLOCK_HIST*2] = bottom_histograms[j+BLOCK_HIST*2];
            bottom_histograms[j+BLOCK_HIST*2] = 0;
         }
         if (wBlk>0 && wBlk<=(((unsigned)(cols +  CELL_SIZE - 1)) / (CELL_SIZE))) wCellBin=0;
         ++wBlk;
      }

      /* The out_histograms is grouped by 8 short variables and put into the channel.
       * Hence it cost 5 loops to complete transmitting the 36 short variables. */
      if (row >=  CELL_SIZE && row/CELL_SIZE<=rows/CELL_SIZE && wCellBin<5) {
             ushort8 res;
             res.s0= out_histograms[0]>>AMP_Norm;
             res.s1= out_histograms[1]>>AMP_Norm;
             res.s2= out_histograms[2]>>AMP_Norm;
             res.s3= out_histograms[3]>>AMP_Norm;
             res.s4= out_histograms[4]>>AMP_Norm;
             res.s5= out_histograms[5]>>AMP_Norm;
             res.s6= out_histograms[6]>>AMP_Norm;
             res.s7= out_histograms[7]>>AMP_Norm;
             write_channel_altera(chHist_Norm, res);
             wCellBin++;
      }
      /* Update the out_histograms by shifting out written variables with new variables to be written. */
      #pragma unroll
      for (int j = 0; j < BLOCK_HIST-8; j++)
         out_histograms[j]=out_histograms[j+8];
      #pragma unroll
      for (int j=BLOCK_HIST-8;j<BLOCK_HIST;++j) out_histograms[j]=0;

      ++offset;
      if (offset==0x10)  {
    	  offset=0;
    	  ++block_count;
    	  base+=16;
    	  block_count=base >= MAXCOLS ? 0 : block_count;
          wBlk       =base >= MAXCOLS ? 0 : wBlk;
    	  base=(base>=MAXCOLS)?0:base;
      }
      offset &= 0x0F;

      row_in_cell = base == 0 && offset == 0 ? row_in_cell + 1 : row_in_cell;
      row_in_cell &= 0x07;
      row = base == 0 && offset == 0 ? row + 1 : row;
    }
}

#define AMP_MAXWEIGHT 10000.0f /*!< The scalor using to convert from floating point to integer. */
#define NORM_MAXWEIGHT 2000.0f /*!< The maximum weight value for normalized weight. */
/*! @brief Normalize each block based on L2-norm algorithm.
 * @details For better invariance to illumination, shadowing etc.,
 * the normalizeit() kernel accumulates a measure of local histogram energy over blocks and uses the results to normalize all of the cells in the block.
 * It uses the 2 rounds L2-norm formular. That is v=v/sqrt( v^2 + delta ).
 * @note It corresponds to the gpu::HOGCache::normalizeBlockHistogram() of OpenCV library.
 * @param[in] rows : The resized rows.
 * @param[in] cols : The resized columes.
 * @param[in] pixels : The total pixels number to read.
 * @param[in] pixwrite : The total pixels number to be written.
 */
void kernel normalizeit(int rows, int cols, int pixels, int pixwrite) {
   /* Three buffers are created 
    * The cbuf is the channel buffer which pops up the histograms data from the channel, 8 short varialbe per loop.
    * The ibuf is the input buffer for normalization, reading one short variable per loop and calucating the 1st round normalization.
    * The obuf is the output buffer, reading from the ibuf and calculating the 2nd round normalization, which is the final result. */
   ushort cbuf[8]; ushort8 input;
   ushort ibuf[BLOCK_HIST];
   ushort obuf[BLOCK_HIST];
   int hist=0, psum1=0,sum1=0, psum2=0, sum2=0, iWrt=0;
   ushort res1=0, res2=0;

   /* In each iteration, one bin data is processed in ibuf and outbuf respectively. */
   for (int i=0; i<pixels; i++) {    
       /* It reads a set of short variable each time from the channel, and distributes them one per iteration. */
       if ((hist&0x7)==0 && iWrt<pixwrite ) {
           input=read_channel_altera(chHist_Norm);
           cbuf[0]=input.s0; cbuf[1]=input.s1; cbuf[2]=input.s2; cbuf[3]=input.s3;
           cbuf[4]=input.s4; cbuf[5]=input.s5; cbuf[6]=input.s6; cbuf[7]=input.s7;
       }
       res1=ibuf[BLOCK_HIST-1];
       res2=obuf[BLOCK_HIST-1];
       /* Calculate 2nd round normalized result, which is stored for the next kernel svm(). */
       if (iWrt>=BLOCK_HIST*2)  {
           write_channel_altera(chNorm_Svm, (ushort)ldexp( res2/(sqrt(sum2)+0.1f) , AMP_Svm));
       }
       /* After finishing calculating the current data, two buffers shift the old data out and fill with new data. */
       #pragma unroll
       for (int j=BLOCK_HIST-2;j>=0;--j) {
           ibuf[j+1]=ibuf[j]; obuf[j+1]=obuf[j];
       }
       /* Calculating the 1st round normalized result, which is stored into the obuf. */
       obuf[0]=min(AMP_MAXWEIGHT/ (sqrt(sum1) + (NBINS * BLOCK_SIZE * BLOCK_SIZE * 0.1f))*res1, NORM_MAXWEIGHT);
       ibuf[0]=cbuf[0];
       ++hist;
       psum1+=(ibuf[0]&0xffff)*(ibuf[0]&0xffff);
       psum2+=(obuf[0]&0xffff)*(obuf[0]&0xffff);
       if (hist==BLOCK_HIST)  {
           hist=0;
           sum1=psum1; psum1=0;
           sum2=psum2; psum2=0;
       }
       /* The cbuf shifts out the read data and put the next-to-read data in the front. */
       #pragma unroll
       for (int j=0;j<7;++j) cbuf[j]=cbuf[j+1];
       ++iWrt;
   }
}
