load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';

DScale = 500000;
Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

noOfEQs = length(Latitude);
cubeLenL = 0.15/10;
cubeLenD = 0.025/10;

for ii = 1:noOfEQs
    
    LongitudeS = Longitude(ii);
    LatitudeS = Latitude(ii);
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
    for ii = 2:170
        
        cThisFrame = cThis(:,:,ii);
        
        thisInd1 = isnan(cThisFrame);
        cThisFrameNoNaN = cThisFrame((~thisInd1)&ind2);
        thisDate = dateAll(ii);
        thisDate = datenum(num2str(thisDate),'yyyymmdd');
        if abs(thisDate-Date0S)<50
            thisTriplet = [lon2 lat2 (single(thisDate)*ones(size(lon2))) cThisFrameNoNaN];
            allTriplets = [thisTriplet; allTriplets];
        end
    end
end 
    save demo44Triplets allTriplets
    noOfPoints = size(allTriplets,1);
    for ii =1:noOfPoints
    end

    
