close all;
clear all;
load  UKEQs2016-2018 

worldmap([49 59],[-12 4]) 
geoshow('landareas.shp') 

DScale = 500000;
 Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

scaleD0 = (thisDate0 - min(thisDate0(:)))./(max(thisDate0(:))-min(thisDate0(:)));
hold on;
view(3)
stem3m( Latitude0,Longitude0, round(scaleD0*DScale),'r+','LineStyle','none');
scaleD1 = (thisDate1 - min(thisDate1(:)))./(max(thisDate1(:))-min(thisDate1(:)));
stem3m( Latitude1,Longitude1, round(scaleD1*DScale),'b+','LineStyle','none');

cubeLenL = 0.15;
cubeLenD = 0.025;
Longitude0S = Longitude0(40);
Latitude0S = Latitude0(40);
scaleD0S = scaleD0(40);
plot3m([Latitude0S+cubeLenL Latitude0S-cubeLenL], [Longitude0S-cubeLenL Longitude0S-cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S-cubeLenD)*DScale], 'k');
plot3m([Latitude0S-cubeLenL Latitude0S-cubeLenL], [Longitude0S+cubeLenL Longitude0S-cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S-cubeLenD)*DScale], 'k');
plot3m([Latitude0S+cubeLenL Latitude0S+cubeLenL], [Longitude0S-cubeLenL Longitude0S+cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S-cubeLenD)*DScale], 'k');
plot3m([Latitude0S-cubeLenL Latitude0S+cubeLenL], [Longitude0S+cubeLenL Longitude0S+cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S-cubeLenD)*DScale], 'k');

plot3m([Latitude0S+cubeLenL Latitude0S-cubeLenL], [Longitude0S-cubeLenL Longitude0S-cubeLenL], [(scaleD0S+cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S-cubeLenL Latitude0S-cubeLenL], [Longitude0S+cubeLenL Longitude0S-cubeLenL], [(scaleD0S+cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S+cubeLenL Latitude0S+cubeLenL], [Longitude0S-cubeLenL Longitude0S+cubeLenL], [(scaleD0S+cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S-cubeLenL Latitude0S+cubeLenL], [Longitude0S+cubeLenL Longitude0S+cubeLenL], [(scaleD0S+cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');

plot3m([Latitude0S-cubeLenL Latitude0S-cubeLenL], [Longitude0S-cubeLenL Longitude0S-cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S-cubeLenL Latitude0S-cubeLenL], [Longitude0S+cubeLenL Longitude0S+cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S+cubeLenL Latitude0S+cubeLenL], [Longitude0S-cubeLenL Longitude0S-cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
plot3m([Latitude0S+cubeLenL Latitude0S+cubeLenL], [Longitude0S+cubeLenL Longitude0S+cubeLenL], [(scaleD0S-cubeLenD)*DScale (scaleD0S+cubeLenD)*DScale], 'k');
ax = gca;
ax.Clipping = 'off';
ind = 0;

for ii = 0.03:0.03:4
    ind = ind +1;
    %view([45 20]);
    p =  1.0e+06 * [    0.0264    5.9465    0.2574 ];
    cameraPosition = [Latitude0S Longitude0S scaleD0S*DScale]; 
    cameraTarget = [Latitude0S/10, Longitude0S/10, scaleD0S*DScale/10]; 
    %cameraAngle = 5.57428;
    %set(gca, 'CameraTarget', cameraTarget);
    %set(gca,'CameraPosition', cameraPosition, 'CameraTarget', cameraTarget);
    
    %camposm(Latitude0S*(2-ii), Longitude0S*(2-ii), scaleD0S*DScale*(2-ii));
    
    camtargm(Latitude0S, Longitude0S, scaleD0S*DScale);
    zoom(1.04);
    drawnow;
    F(ind) = getframe(gcf);
    
    pause(0.01);
end

Video = VideoWriter('output2.avi');
open(Video);
writeVideo(Video,F);
close(Video);

