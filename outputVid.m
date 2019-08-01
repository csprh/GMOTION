load F;

ind0 = 0;
lenF = length(F);
for ii = 1:1:lenF
    thisF = F(ii);
    ii
    if size(thisF.cdata,1) == 0 
        continue
    end
    ind0 = ind0 +1;
    outF(ind0) = thisF;
end






Video = VideoWriter('output3.avi');
Video.FrameRate=15;
open(Video);
writeVideo(Video,outF);
close(Video);