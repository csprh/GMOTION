load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';

startL = [1 1]; countL = [68941 18482];
startLC = [startL 100]; countLC = [countL 100];

lonThis = h5read(frameName,'/Longitude', startL, countL);
latThis = h5read(frameName,'/Latitude', startL, countL);
    
    
cThis = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
    cThis_100 = cThis(:,:,100);
    thisInd2 = isnan(cThis);
    lon2 = lonThis((~thisInd2));
    lat2 = latThis((~thisInd2));
    lon2 = lon2(:);
    lat2 = lat2(:);
    
    save latLons lat2 lon2;
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
