load stokeData;
for ii = 1:169
    scatter3(lat2,lon2,p2(:,ii)-mean(p2(:,ii)),'.');
    set(gca,'ZLim',[-200 200])
    pause(0.5);
end