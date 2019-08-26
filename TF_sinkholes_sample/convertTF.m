clear all;
close all;
S = shaperead('TF_sinkholes_sample');

Slen = length(S);
ind = 1;
for ii= 1: Slen;

    S(ii).Width_m
    thisLat(ii) = S(ii).X_long;
    thisLon(ii) = S(ii).Y_lat;
    thisDate(ii) = datenum(S(ii).Date,'yyyymmdd');

    try
        dpthTmp = str2num(S().Depth_m);
        if dpthTmp < 1
            continue
        end
        thisDepth(ind) = dpthTmp;
         
    catch
        thisDepth(ind) = 0;
    end
    
    try
        thisWidth(ind) = str2num(S(ii).Width_m);
    catch
        thisWidth(ind) = 0;
    end
    ind = ind + 1;
end
hist(thisDepth);

