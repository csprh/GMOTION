function [thisMax, thisMin] = getMinMaxFromH5s(filenameBase)
%% This Code loops through al the H5 output files and generates
%% The maximum and minimum values for each modality.
%% These are then output into the maximum vector thisMax and
%% the minimum vector thisMin
% USAGE:
%   [thisMax, thisMin] = getMinMaxFromH5s(filenameBase)
% INPUT:
%   filenameBase: Directory that holds all compressed h5 datacubes
% OUTPUT:
%   thisMax: Vector of maximum values for each modality
%   thisMin: Vector of minimum values for each modality

% THE UNIVERSITY OF BRISTOL: HAB PROJECT
% Author Dr Paul Hill March 2019
close all;

h5files=dir([filenameBase '*.h5.gz']);
numberOfH5s=size(h5files,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Loop through all the ground truth entries%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii = 1: numberOfH5s
    ii
    try
        %% Process input h5 file
        system(['rm ' filenameBase '*.h5']);
        gzh5name = [filenameBase h5files(ii).name];
        gunzip(gzh5name);
        h5name = gzh5name(1:end-3);
        
        if ii == 1
            % Initalise
            thisMax = ones(numberOfH5s)*NaN;
            thisMin = ones(numberOfH5s)*NaN;
        end
        PointsProj = h5read(h5name, '/PointsProj');
        theseVals = PointsProj(:,3);
        thisMax(ii) = max(theseVals);
        thisMin(ii) = min(theseVals);
        
    catch
    end
end



