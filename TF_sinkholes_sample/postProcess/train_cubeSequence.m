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
tmpStruct = xml2struct('configSHUnderDesk.xml');

cubesDir = tmpStruct.confgData.trainDir.Text;
imsDir = tmpStruct.confgData.trainImsDir.Text;

preLoadMinMax = str2num(tmpStruct.confgData.preLoadMinMax.Text);
threshBytes = str2num(tmpStruct.confgData.threshBytes.Text);
noOfIms = str2num(tmpStruct.confgData.noOfIms.Text);


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

thisInd = 1;
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
        thisDepth = h5readatt(h5name,'/GroundTruth/','thisDepth');
        
        [ 'thisWidth = ' num2str(thisDepth) ];
        isSH  = thisDepth > 0;
        
        % Discount this line in the Ground Truth
        if dirOut.bytes < threshBytes
            totalDiscount= totalDiscount+1;
            totalDiscount
            continue;
        end
        
        %Split output into train/test, HAB Class directory, Ground truth line
        %number, Group Index
        
        if isSH == 1
            for rotflip = 1:4  
                baseDirectory = [ imsDir filesep num2str(isSH) filesep num2str(thisInd)] ;
                outputImagesFromDataCube(baseDirectory,   groupMinMax,  h5name, noOfIms, rotflip);
                thisInd = thisInd + 1;
            end
        else
                baseDirectory = [ imsDir filesep num2str(isSH) filesep num2str(thisInd)] ;
                outputImagesFromDataCube(baseDirectory,   groupMinMax,  h5name, noOfIms, 1);
                thisInd = thisInd + 1;  
        end
        
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


