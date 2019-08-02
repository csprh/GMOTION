load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';

DScale = 500000;
Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

noOfEQs = length(Latitude);
cubeLenL = 0.15/10;
cubeLenD = 0.025/10;
dateDelta = 50;

for ii = 1:noOfEQs
    
    LongitudeS = Longitude(ii);
    LatitudeS = Latitude(ii);
    zone = utmzone(thisLat, LongitudeS);
    utmstruct = defaultm('utm');
    utmstruct.zone = zone;
    utmstruct.geoid = wgs84Ellipsoid; %almanac('earth','grs80','meters');
    utmstruct = defaultm(utmstruct);
    [centerXProj, centerYProj] = mfwdtran( utmstruct, LatitudeS,LongitudeS);
    DateS = thisDate(ii);
    MLS = ML(ii);
    
    lat0 = LatitudeS-cubeLenL; lat1 = LatitudeS+cubeLenL;
    lon0 = LongitudeS-cubeLenL; lon1 = LongitudeS+cubeLenL;
    
    lonAll = h5read(frameName,'/Longitude');
    latAll = h5read(frameName,'/Latitude');
    
    ind2 = ((latAll>lat0)&(latAll<lat1)&(lonAll>lon0)&(lonAll<lon1));
    
    clear latAll; clear lonAll;
    
    dateAll = h5read(frameName,'/Date');
    
    [a, b] = find(ind2);
    minIndLon = min(a);maxIndLon = max(a);
    
    minIndLat = min(b);maxIndLat = max(b);
    
    startL = [minIndLon minIndLat]; countL = [maxIndLon-minIndLon maxIndLat-minIndLat];
    startLC = [startL 1]; countLC = [countL 170];
    lonThis = h5read(frameName,'/Longitude', startL, countL);
    latThis = h5read(frameName,'/Latitude', startL, countL);
    
    
    cThis = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
    cThis_100 = cThis(:,:,100);
    thisInd2 = isnan(cThis_100);
    ind2 = ((latThis>lat0)&(latThis<lat1)&(lonThis>lon0)&(lonThis<lon1));
    lon2 = lonThis((~thisInd2)&ind2);
    lat2 = latThis((~thisInd2)&ind2);
    
    allTriplets  = [];
    

    [xProj xProj] = mfwdtran( utmstruct, lat2,lon2);
    xProjDelta = xProj - LongitudeS;
    yProjDelta = yProj - LatitudeS;
    for iii = 2:170
        
        cThisFrame = cThis(:,:,iii);
        
        thisInd1 = isnan(cThisFrame);
        cThisFrameNoNaN = cThisFrame((~thisInd1)&ind2);
        thisDate = dateAll(iii);
        thisDate = datenum(num2str(thisDate),'yyyymmdd');
        
        
        if (thisDate-DateS)<dateDeltaThresh
            dateChange = thisDate-DateS;

            thisTriplet = [xProjDelta yProjDelta (single(dateChange)*ones(size(xProjDelta))) cThisFrameNoNaN];
            allTriplets = [thisTriplet; allTriplets];
        end
    end
end 

function addToH5(h5name, , allTriplets)
fid = H5F.create(h5name);
plist = 'H5P_DEFAULT';
gid = H5G.create(fid,'GroundTruth',plist,plist,plist);
H5G.close(gid);
H5F.close(fid);

h5writeatt(inStruc.h5name,'/GroundTruth', 'thisLat', thisLat);
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisLon', thisLon);
h5writeatt(inStruc.h5name,'/GroundTruth', 'thisCount', thisDate);
h5writeatt(inStruc.h5name,'/GroundTruth', 'dayEnd', thisML);

hdf5write(h5name,['/' thisMod  '/Ims'],theseImages, 'WriteMode','append');
hdf5write(h5name,['/' thisMod  '/theseDates'],theseDates, 'WriteMode','append');
hdf5write(h5name,['/' thisMod  '/theseDeltaDates'],theseDeltaDates, 'WriteMode','append');
hdf5write(h5name,['/' thisMod  '/Points'],thesePointsOutput, 'WriteMode','append');
hdf5write(h5name,['/' thisMod  '/PointsProj'],thesePointsOutputProj, 'WriteMode','append');
end
