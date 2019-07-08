load  UKEQs2016-2018 

worldmap([49 59],[-12 4]) 
geoshow('landareas.shp') 

Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

scaleD = (thisDate0 - min(thisDate0(:)))./(max(thisDate0(:))-min(thisDate0(:)));
hold on;
view(3)
stem3m( Latitude0,Longitude0, round(scaleD*500000),'r+','LineStyle','none');
scaleD = (thisDate1 - min(thisDate1(:)))./(max(thisDate1(:))-min(thisDate1(:)));
stem3m( Latitude1,Longitude1, round(scaleD*500000),'b+','LineStyle','none');
scatter3m(Longitude0, Latitude0, thisDate0, ones(size(Latitude0)), ones(size(Latitude0)) );
Latitude0;
