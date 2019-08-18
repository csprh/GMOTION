function getAllCubes

load  UKEQs2016-2018
%frameName = '../030A_03647_101313-vel.h5';


DScale = 500000;
Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

noOfEQs = length(Latitude);

tmpStruct = xml2struct('configEQUnderDesk.xml');

confgData.outDir = tmpStruct.confgData.trainDir.Text;
confgData.distance1 = str2double(tmpStruct.confgData.distance1.Text);
confgData.resolution = str2double(tmpStruct.confgData.resolution.Text);
confgData.numberOfDaysInPast = str2double(tmpStruct.confgData.numberOfDaysInPast.Text);
confgData.cubeLenL = str2double(tmpStruct.confgData.cubeLenL.Text);
confgData.frameName = tmpStruct.confgData.frameName.Text;

system(['rm ' confgData.outDir '*.h5']);

for ii = 11:noOfEQs
    
    fileName = ['Cube_' sprintf('%05d',ii) '_' sprintf('%05d',ii) '_' num2str(thisDate(ii)) '.h5'];
    
    inStruc.h5name = [confgData.outDir fileName];
    
    inStruc.thisLon = Longitude(ii);
    inStruc.thisLat = Latitude(ii);
    zone = utmzone(inStruc.thisLat, inStruc.thisLon);
    utmstruct = defaultm('utm');
    utmstruct.zone = zone;
    utmstruct.geoid = wgs84Ellipsoid; %almanac('earth','grs80','meters');
    utmstruct = defaultm(utmstruct);

    inStruc.dayEnd = thisDate(ii);
    inStruc.dayStart = inStruc.dayEnd-confgData.numberOfDaysInPast;
    inStruc.thisML = ML(ii);
    
    addDataH5(inStruc, confgData);
    out = interpRegion(confgData.frameName, confgData.cubeLenL, inStruc.thisLat, inStruc.thisLon);
    
    theseDates = out.interpDates;
    
    datesInd = (theseDates<inStruc.dayEnd)&(theseDates>inStruc.dayStart);
    datesIndNums = 1:length(theseDates);
    dateRange = theseDates(datesInd);
    datesIndNumsRange = datesIndNums(datesInd);
    
    listLength = length(dateRange);
    clear theseDates theseDeltaDates theseImages;
    thesePointsOutput = []; thesePointsProjOutput = [];
    theseImages = cell(listLength,1);theseDates = cell(listLength,1);theseDeltaDates = cell(listLength,1);
    thesePointsNew = cell(listLength,1);
    thesePointsProjNew = cell(listLength,1);
    
    for iii = 1:listLength
        thisDateD = dateRange(iii);
        thisDeltaDate = inStruc.dayEnd-thisDateD;
        theseDates{iii} = thisDateD;
        theseDeltaDates{iii} = thisDeltaDate;
        [theseImages{iii}, thesePoints, thesePointsProj] = getData(out, datesIndNumsRange(iii), inStruc.thisLat, inStruc.thisLon, confgData.distance1, confgData.resolution, utmstruct);        %         thesePointsNew{iii} = [thesePoints ones(size(thesePoints,1),1)*theseDeltaDates{iii}];
        thesePointsNew{iii} = [thesePoints ones(size(thesePoints,1),1)*theseDeltaDates{iii}];
        thesePointsProjNew{iii} = [thesePointsProj ones(size(thesePointsProj,1),1)*theseDeltaDates{iii}];
    end
    %Get rid of any empty cells
    theseDates = theseDates(~cellfun('isempty', theseImages));
    theseDeltaDates = theseDeltaDates(~cellfun('isempty', theseImages));
    theseImages = theseImages(~cellfun('isempty', theseImages));
    %Build output arrays
    for iii = 1:listLength
        thesePointsOutput = [thesePointsOutput; thesePointsNew{iii}];
        thesePointsProjOutput = [thesePointsProjOutput; thesePointsProjNew{iii}];
    end
    
    addToH5(inStruc.h5name, theseImages, theseDates, theseDeltaDates, thesePointsOutput, thesePointsProjOutput);
    gzip(inStruc.h5name);
    system(['rm ' confgData.outDir '*.h5']);
end

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
hdf5write(h5name,'/Ims',theseImages, 'WriteMode','append');
hdf5write(h5name,'/theseDates',theseDates, 'WriteMode','append');
hdf5write(h5name,'/theseDeltaDates',theseDeltaDates, 'WriteMode','append');
hdf5write(h5name,'/Points',thesePointsOutput, 'WriteMode','append');
hdf5write(h5name,'/PointsProj',thesePointsOutputProj, 'WriteMode','append');

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
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisML', inStruc.thisML);
h5writeatt(inStruc.h5name,'/GroundTruth', 'dayEnd', inStruc.dayEnd);
h5writeatt(inStruc.h5name,'/GroundTruth', 'dayStart', inStruc.dayStart);
h5writeatt(inStruc.h5name,'/GroundTruth', 'resolution', confgData.resolution);
h5writeatt(inStruc.h5name,'/GroundTruth', 'distance1', confgData.distance1);


