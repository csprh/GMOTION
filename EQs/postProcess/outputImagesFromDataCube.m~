function outputImagesFromDataCube(baseDirectory,   groupMinMax,  h5name)
%% This code generates quantised images for an input H5 datacube
%% It loops through all modalities within the given H5 file (h5name) and generates
%% A directory of images in a folder for ingress into Machine Learning

% USAGE:
%   outputImagesFromDataCube(baseDirectory,  numberOfDays, groupMinMax, inputRangeX, inputRangeY, alphaSize, outputRes, h5name)
% INPUT:
%   baseDirectory: Directory to put the output images (0,1,2....directories
%   created to put modalities into...each image 1.png, 2.png etc are the
%   days output)
%   numberOfDays: Number of days in temporal range of datacube
%   groupMinMax: Array of Minima and Maxima of the modalities
%   inputRangeX: Range of output for images ([0:50])
%   inputRangeY: Range of output for images ([0:50])
%   alphaSize: Control of resampling
%   outputRes: Resolution (in metres) of quantise bins output
%   h5name: Name of the input H5 file
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: 
% Author Dr Paul Hill July 2019

addpath('Inpaint_nans');

thisBaseDirectory = [baseDirectory '/'];
mkdir(thisBaseDirectory);

Ims = h5read(h5name, '/Ims');

%%Loop through days, quantise them, sum, clip and output
for thisDay  = 1:size(Ims,3)
    try
        outputImage = Ims(:,:,thisDay);
        outputImage(outputImage==0) = NaN;
        outputImage  = inpaint_nans(outputImage );
        thisMin = groupMinMax(1);   thisMax = groupMinMax(2);
        
        outputImage = outputImage-thisMin;
        outputImage = round(255.*(outputImage./(thisMax-thisMin)));
        outputImage(outputImage < 0) = 0; outputImage(outputImage > 255) = 255;
        
        imwrite(uint8(outputImage),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
    catch
        outputImage = ones(size(output.xq))*NaN;
        imwrite(uint8(outputImage),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
    end
end



