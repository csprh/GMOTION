clear all;
close all;

latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];

zone = utmzone(mean(latLims),mean(lonLims));
utmstruct = defaultm('utm');
utmstruct.zone = zone;
utmstruct.geoid = wgs84Ellipsoid; %almanac('earth','grs80','meters');
utmstruct = defaultm(utmstruct);
datLims = [datenum('11-May-2015') datenum('27-Dec-2018')];
worldmap([49 59],[-12 4]); 
geoshow('landareas.shp'); 
gebcoFilename = '/Users/csprh/seadas-7.4/DATA/GEBCO.nc';
S = shaperead('TF_sinkholes_sample', 'UseGeoCoords', true);

%pp = [52.667525,-2.377930;
    
%BNG = [374539 307824;
Slen = length(S);
ind = 1;
for ii= 1: Slen;

    S(ii).Width_m
    

    yy = S(ii).Lat;
    xx = S(ii).Lon;
    

    [tmpLat,tmpLon]= os2llPRH( xx, yy);

    %[tmpLat,tmpLon]= minvtran(utmstruct,  xx, yy);
    %[tmpLon,tmpLat] = projinv(proj,X,Y);
    S(ii).Date
    tmpDate = datenum(S(ii).Date,'yyyymmdd');

    if tmpDate < (datLims(1) + 60) | tmpDate > datLims(2)
        continue
    end
    try
        dpthTmp = str2num(S(ii).Depth_m);
        if dpthTmp < 0.5
            continue
        end
        thisDepth(ind) = dpthTmp;
         
    catch
        thisDepth(ind) = 10;
    end
    
    try
        thisWidth(ind) = str2num(S(ii).Width_m);
    catch
        thisWidth(ind) = 10;
    end
    thisXY(ind,1) = S(ii).Lon;
    thisXY(ind,2) = S(ii).Lat;
    thisLat(ind) = tmpLat;
    thisLon(ind) = tmpLon;
    thisDate(ind) = tmpDate;
    thisURL{ind} = S(ii).Image_URL;
    ind = ind + 1;
end

%load SinkHolesLL

%thisLat = Lat';
%thisLon = Lon';

numOfPos = length(thisWidth);

numOfNeg = numOfPos;

lon1D = ncread(gebcoFilename, '/lon'); 
lat1D = ncread(gebcoFilename, '/lat');
distThresh = 50;
datThresh = 50;
ii = 0;
while ii < numOfNeg
  % generate random position
  % 
    ranLon = rand(1,1)*(lonLims(2)-lonLims(1)) + lonLims(1);
    ranLat = rand(1,1)*(latLims(2)-latLims(1)) + latLims(1);
    ranDat = rand(1,1)*(datLims(2)-datLims(1)) + datLims(1);

    
    [arclen,az] = distance(ranLat,ranLon,thisLat,thisLon);
    distkm = distdim(arclen,'deg','km');
    [thisMinDist ,thisMinIndx] = min(distkm);
    thisMinDat = abs(thisDate(thisMinIndx)-ranDat);
    
    if (thisMinDist< distThresh) && (thisMinDat < datThresh);
        continue;
    end
    
    [~, centre_col] = min(abs(lon1D-ranLon));
    [~, centre_row] = min(abs(lat1D-ranLat));
    bathAt = ncread(gebcoFilename, '/elevation', [centre_col centre_row], [1 1]);
    isSea = bathAt <0;
    if isSea 
        continue;
    end
    ii = ii+1;
    thisLon(ii+numOfPos) = ranLon;
    thisLat(ii+numOfPos) = ranLat;
    thisDate(ii+numOfPos) = round(ranDat);
    thisWidth(ii+numOfPos) = 0;
    thisDepth(ii+numOfPos) = 0;
end


for ii = 1: length(thisDepth)
    outLon = thisLon(ii);
    outLat = thisLat(ii);
    if thisWidth(ii) == 0
        plotm(outLat,outLon,'+r');
    else
        plotm(outLat,outLon,'+b');
    end
    pause(0.01);
end


save UKTFSH2016-2018-RAND thisLon thisLat thisDate thisDepth thisWidth

