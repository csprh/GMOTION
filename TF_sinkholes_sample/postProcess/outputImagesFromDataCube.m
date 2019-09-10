function outputImagesFromDataCube(baseDirectory,   groupMinMax,  h5name, noOfIms, rotflip)
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
%   rotFlip: 1,2,3,4 rotation, flip, none or both
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: 
% Author Dr Paul Hill July 2019

addpath('Inpaint_nans');

thisBaseDirectory = [baseDirectory '/'];


Ims = h5read(h5name, '/Ims');

if size(Ims,3) ~= noOfIms
    return;
end
mkdir(thisBaseDirectory);
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
        
        if rotflip == 1
            imwrite(uint8(outputImage),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
        elseif rotflip == 2
            imwrite(uint8(fliplr(outputImage)),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
        elseif rotflip ==3
            imwrite(uint8(flipud(outputImage)),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
        elseif rotflip == 4
            imwrite(uint8(rot90(outputImage,2)),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
        end
        
    catch
        outputImage = ones(size(output.xq))*NaN;
        imwrite(uint8(outputImage),[thisBaseDirectory  sprintf('%02d',thisDay),'.png']);
    end
end



