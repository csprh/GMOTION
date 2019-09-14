function getAllCubes

load  UKTFSH2016-2018
thisDate= round(thisDate);
%frameName = '../030A_03647_101313-vel.h5';

DScale = 500000;
Latitude0 = thisLat(thisDepth==0); Longitude0 = thisLon(thisDepth==0); thisDate0 = thisDate(thisDepth==0);
Latitude1 = thisLat(thisDepth~=0); Longitude1 = thisLon(thisDepth~=0); thisDate1 = thisDate(thisDepth~=0);

noOfSHs = length(thisLon);

%tmpStruct = xml2struct('configSHUnderDeskRural.xml');
tmpStruct = xml2struct('configSHUnderDesk.xml');

confgData.outDir = tmpStruct.confgData.trainDir.Text;
confgData.distance1 = str2double(tmpStruct.confgData.distance1.Text);
confgData.resolution = str2double(tmpStruct.confgData.resolution.Text);
confgData.numberOfDaysInPast = str2double(tmpStruct.confgData.numberOfDaysInPast.Text);
confgData.cubeLenL = str2double(tmpStruct.confgData.cubeLenL.Text);
confgData.frameName = tmpStruct.confgData.frameName.Text;

system(['rm ' confgData.outDir '*.h5']);

stats.NoOfIms = 0;

for ii = 1:noOfSHs
    ii
    fileName = ['Cube_' sprintf('%05d',ii) '_' sprintf('%05d',ii) '_' num2str(thisDate(ii)) '.h5'];
    
    inStruc.h5name = [confgData.outDir fileName];
    
    inStruc.thisLon = thisLon(ii);
    inStruc.thisLat = thisLat(ii);
    zone = utmzone(inStruc.thisLat, inStruc.thisLon);
    utmstruct = defaultm('utm');
    utmstruct.zone = zone;
    utmstruct.geoid = wgs84Ellipsoid; %almanac('earth','grs80','meters');
    utmstruct = defaultm(utmstruct);
    
    inStruc.dayEnd = thisDate(ii);
    inStruc.dayStart = inStruc.dayEnd-confgData.numberOfDaysInPast;
    inStruc.thisDepth = thisDepth(ii);
    
    
    try
        out = interpRegion(confgData.frameName, confgData.cubeLenL, inStruc.thisLat, inStruc.thisLon);
    catch
        continue;
    end
    if length(out.outcdTSmooth) == 0
        continue
    end
    addDataH5(inStruc, confgData);
    theseDates = out.interpDates;
    
    datesInd = (theseDates<inStruc.dayEnd)&(theseDates>inStruc.dayStart);
    datesIndNums = 1:length(theseDates);
    dateRange = theseDates(datesInd);
    datesIndNumsRange = datesIndNums(datesInd);
    
    listLength = length(dateRange);
    clear theseDates theseDeltaDates theseImages;
    
    if listLength == 0
        continue;
    end
    iii = 1;

    [theseImages, thesePoints, thesePointsProj] = getDataStats(out, datesIndNumsRange(iii), inStruc.thisLat, inStruc.thisLon, confgData.distance1, confgData.resolution, utmstruct);        %         thesePointsNew{iii} = [thesePoints ones(size(thesePoints,1),1)*theseDeltaDates{iii}];
    
    stats.NoOfIms = stats.NoOfIms + 1;
    stats.SizeOfImsW = size(theseImages,1);
    stats.SizeOfImsH = size(theseImages,2);
    theseImages = theseImages(:);
    stats.NoOfZerosInIm(stats.NoOfIms) = sum(theseImages==0);
    stats.NoOfNonZerosInIm(stats.NoOfIms) = sum(theseImages~=0);
    stats.NoOfPoints(stats.NoOfIms) = size(thesePoints,1);
    stats.NoOfPointsProj(stats.NoOfIms) = size(thesePointsProj,1);
end

save stats stats

function addToH5(h5name,  theseImages, theseDates, theseDeltaDates, thesePointsOutput, thesePointsOutputProj)
%% add Ims, theseDates, theseDeltaDates and Points to output H5 file
%
% USAGE:
%   addToH5(h5name, thisMod, theseImages, theseDates, theseDeltaDates, thesePointsOutput, thesePointsOutputProj)
% INPUT:
%   h5name - name of H5 name to be output
%   thisMod = Name of the output modality
%   theseImages - Cell array of output binned images (for this modality)
%   theseDates - The actual capture dates of the points and images output
%   theseDeltaDates - The delta dates (difference from capture date) of the points and images output
%   thesePointsOutput - 4D Array of points output
%   thesePointsOutputProj - 4D Array of projected points output
% OUTPUT:
%   -
try
    hdf5write(h5name,'/Ims',theseImages, 'WriteMode','append');
    hdf5write(h5name,'/theseDates',theseDates, 'WriteMode','append');
    hdf5write(h5name,'/theseDeltaDates',theseDeltaDates, 'WriteMode','append');
    hdf5write(h5name,'/Points',thesePointsOutput, 'WriteMode','append');
    hdf5write(h5name,'/PointsProj',thesePointsOutputProj, 'WriteMode','append');
catch
end

function addDataH5(inStruc, confgData)
%% addDataH5 creates a H5 file and stores ground truth data to it
%  Adds extracted information to one H5 file per datapoint in ground truth
%
% USAGE:
%   addDataH5(inStruc, confgData)
% INPUT:
%   inStruc - Contains all the input parameters for the function
%   confgData - Configuration information extracted from XML
% OUTPUT:
%   -

fid = H5F.create(inStruc.h5name);
plist = 'H5P_DEFAULT';
gid = H5G.create(fid,'GroundTruth',plist,plist,plist);
H5G.close(gid);
H5F.close(fid);
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisLat', inStruc.thisLat);
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisLon', inStruc.thisLon);
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisDepth', inStruc.thisDepth);
h5writeatt(inStruc.h5name,'/GroundTruth', 'dayEnd', inStruc.dayEnd);
h5writeatt(inStruc.h5name,'/GroundTruth', 'dayStart', inStruc.dayStart);
h5writeatt(inStruc.h5name,'/GroundTruth', 'resolution', confgData.resolution);
h5writeatt(inStruc.h5name,'/GroundTruth', 'distance1', confgData.distance1);


