close all;
clear all;
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

ind = 0;
for ii = 20:0.5:90
    ind = ind +1;
    view([45 ii]);
    drawnow;
    F(ind) = getframe(gcf);
    pause(0.01);
end

Video = VideoWriter('output.avi');
open(Video);
writeVideo(Video,F);
close(Video);




