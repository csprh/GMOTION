function train_cubeSequence
%% This Code loops through all the h5 output files and generates
%% A directory of images in a folder for ingress into Machine Learning model.
% Datapoints (lines in the ground truth file) are discounted if they do not
% Contain enough data. Using the thresholds in the XML file
% 
% Optionally loops through all H5 datacubes to generate min and max values for
% all modalities

% Tests datacubes to see if there is enough data to discount the training using
% that datacube
% If tests are passed then outputImagesFromDataCube.m is used to generate the
% of quantised images
%
% USAGE:
%   train_cubeSequence;
% INPUT:
%   -
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: DE PROJECT
% Author Dr Paul Hill July 2019
clear; close all;
addpath('..');
tmpStruct = xml2struct('configEQUnderDesk.xml');


cubesDir = tmpStruct.confgData.trainDir.Text;
imsDir = tmpStruct.confgData.trainImsDir.Text;
resolution = str2num(tmpStruct.confgData.resolution.Text);
distance1 = str2num(tmpStruct.confgData.distance1.Text);
outputRes = str2num(tmpStruct.confgData.outputRes.Text);
preLoadMinMax = str2num(tmpStruct.confgData.preLoadMinMax.Text);
numberOfDaysInPast  = str2num(tmpStruct.confgData.numberOfDaysInPast.Text);
threshBytes = str2num(tmpStruct.confgData.threshBytes.Text);


%The input range is usually 50 by 50 samples (in projected space)
%The output resolution is 1000m (1km).  This results in 100x100 pixels images
%AlphaSize controls the interpolation projected points to output image

inputRangeX = [0 distance1/resolution];
inputRangeY = [0 distance1/resolution];


h5files=dir([cubesDir '*.h5.gz']);
numberOfH5s=size(h5files,1);

totalDiscount = 0;  %Number of discounted datapoints

if preLoadMinMax ~= 1
    [thisMax, thisMin] = getMinMaxFromH5s(cubesDir);
    groupMinMax = getMinMax(thisMax, thisMin);
    save groupMaxAndMin groupMinMax
else
    load groupMaxAndMin %load the max and minima of the mods
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Loop through all the ground truth entries%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii = 1: numberOfH5s
    try
        %% Process input h5 file
        system(['rm ' cubesDir '*.h5']);
        gzh5name = [cubesDir h5files(ii).name];
        dirOut = dir(gzh5name);
        gunzip(gzh5name);
        h5name = gzh5name(1:end-3);
        thisML = h5readatt(h5name,'/GroundTruth/','thisML');
        
        [ 'thisML = ' num2str(thisML) ];
        isEQ  = thisML > 0;

        
        dirOut = dir(gzh5name);
        
        % Discount this line in the Ground Truth
        if dirOut.bytes < threshBytes
            totalDiscount= totalDiscount+1;
            totalDiscount
            continue;
        end
        
        %Split output into train/test, HAB Class directory, Ground truth line
        %number, Group Index
        baseDirectory = [ imsDir filesep num2str(isEQ) '/' num2str(ii)] ;
        
        outputImagesFromDataCube(baseDirectory,  numberOfDaysInPast, groupMinMax, inputRangeX, inputRangeY, alphaSize, outputRes, h5name);
        
        clear totNumberCP zNumberCP quotCP totNumber zNumber quot
    catch
        [ 'caught at = ' num2str(ii) ]
    end
end


function groupMinMax = getMinMax(thisMax, thisMin)
% USAGE:
%   groupMinMax = getMinMax(thisMax, thisMin)
% INPUT:
%   thisMax = array of maxima
%   thisMin = array of minima
% OUTPUT:
%   groupMinMax = group together the minimum and maximum of input min and max
groupMinMax = [ min(thisMin') ; max(thisMax')]';


