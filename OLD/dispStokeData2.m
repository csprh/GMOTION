load stokeData;

writerObj = VideoWriter('ColorStoke4.avi');
writerObj.FrameRate = 3;

open(writerObj);


%hold on;

thisZZ = p2;
for ii = 1:169
    %thisZ = p2(:,ii)-mean(p2(:,ii));
    thisZ = p2(:,ii);
    TempScaled = (thisZ - min(thisZZ(:)))/(max(thisZZ(:))-min(thisZZ(:)));
    TempScaled = (thisZ - min(thisZ(:)))/(max(thisZ(:))-min(thisZ(:)));
    %scatter(lon2,lat2,1, TempScaled);
    %plot_google_map('MapScale', 1, 'APIKey','AIzaSyBQ5Px_1bwuiPH9Nz94jeaWncRtTYcB2m');
    %scatter3(lat2,lon2,p2(:,ii)-mean(p2(:,ii)),'.');
    scatter3(lat2,lon2,p2(:,ii),'.');
    set(gca,'ZLim',[-200 200]);
    frame = getframe(gcf) ;
    writeVideo(writerObj, frame);
    pause(0.01);
end
close(writerObj);
