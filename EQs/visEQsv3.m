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
points0 = stem3m( Latitude0,Longitude0, round(scaleD0*DScale),'r+','LineStyle','none');
scaleD1 = (thisDate1 - min(thisDate1(:)))./(max(thisDate1(:))-min(thisDate1(:)));
points1 = stem3m( Latitude1,Longitude1, round(scaleD1*DScale),'b+','LineStyle','none');

cubeLenL = 0.15/10;
cubeLenD = 0.025/10;
Longitude0S = Longitude0(44);
Latitude0S = Latitude0(44);
scaleD0S = scaleD0(44);
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


for ii = 0.03:0.05:6
    ind = ind +1;
    %view([45 20]);
    p =  1.0e+06 * [    0.0264    5.9465    0.2574 ];
    cameraPosition = [Latitude0S Longitude0S scaleD0S*DScale]; 
 
    %cameraAngle = 5.57428;
    %set(gca, 'CameraTarget', cameraTarget);
    %set(gca,'CameraPosition', cameraPosition, 'CameraTarget', cameraTarget);
    
    %camposm(Latitude0S*(2-ii), Longitude0S*(2-ii), scaleD0S*DScale*(2-ii));
    
    camtargm(Latitude0S, Longitude0S, scaleD0S*DScale);
    zoom(1.06);
    drawnow;
    F(ind) = getframe(gcf);
    
    pause(0.01);
end

set(points0(1),'XData',[], 'YData', [], 'ZData', []);
set(points0(2),'XData',[], 'YData', [], 'ZData', []);
set(points1(1),'XData',[], 'YData', [], 'ZData', []);
set(points1(2),'XData',[], 'YData', [], 'ZData', []);
load demo44Triplets

theseDates = allTriplets(:,3);
theseMotions = allTriplets(:,4);
theseMotions = (theseMotions - min(theseMotions(:)))./(max(theseMotions(:))-min(theseMotions(:)));
scaleD0 = (theseDates - min(thisDate0(:)))./(max(thisDate0(:))-min(thisDate0(:)));
theseColors = jet;
for ii = 1:1:size(allTriplets,1)

    ind = ind +1;
    %view([45 20]);
    %Latitude0 = double(allTriplets(:,2));
    %Longitude0 = double(allTriplets(:,1));
    %scaleD0S2 = 0.306+0.1*double(scaleD0(:)); 
    %scaleD0S2 = 1.7656*double(scaleD0(:)); 
    %thisMotionColor = [theseMotions(:) 0 1-theseMotions(:)];
    %points2 = stem3m( Latitude0,Longitude0, round(scaleD0S2*DScale),  '.g','LineStyle','none'); %'Color', thisMotionColor,
    Latitude0 = double(allTriplets(ii,2));
    Longitude0 = double(allTriplets(ii,1));
    scaleD0S2 = 0.3065+0.1*double(scaleD0(ii));
    thisMotionColor = round(1+theseMotions(ii)*63);
    thisMotionColor = theseColors(thisMotionColor,:);
    %thisMotionColor = [theseMotions(ii) 0 1-theseMotions(ii)];
    points2 = stem3m( Latitude0,Longitude0, round(scaleD0S2*DScale),  '.','LineStyle','none'); %'Color', thisMotionColor,
    set(points2(1),'Color', thisMotionColor);
    set(points2(2),'Color', thisMotionColor);
    %cameraAngle = 5.57428;
    %get(gca, 'CameraTarget', cameraTarget);
    %set(gca,'CameraPosition', cameraPosition, 'CameraTarget', cameraTarget);
    
    %camposm(Latitude0S*(2-ii), Longitude0S*(2-ii), scaleD0S*DScale*(2-ii));
    
    %camtargm(Latitude0S, Longitude0S, scaleD0S*DScale);
    if rem(ii,50)==0
        pause(0.01);
        drawnow;
        F(ind) = getframe(gcf);
        ind = ind +1;
    end
        
end

for ii = 1:1:30
    zoom(1.01);
    drawnow;
    F(ind) = getframe(gcf);
    ind = ind +1;
end






Video = VideoWriter('output3.avi');
open(Video);
writeVideo(Video,F);
close(Video);

