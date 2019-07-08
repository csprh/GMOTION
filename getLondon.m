tl = [51.520250 -0.205083];
br = [51.457298 0.000052];
tl = [53.025981, -2.253819];
br = [52.971963, -2.131686];
lon1 = h5read('030A_03647_101313-vel.h5','/Longitude');
lat1 = h5read('030A_03647_101313-vel.h5','/Latitude');
thisTL = abs(tl(1)-lat1)+abs(tl(2)-lon1);
[min_val,idx]=min(thisTL(:))
[row1,col1]=ind2sub(size(thisTL),idx)
clear thisTL

thisBR = abs(br(1)-lat1)+abs(br(2)-lon1);
[min_val,idx]=min(thisBR(:))
[row2,col2]=ind2sub(size(thisBR),idx)
clear thisBR

thisTR = abs(tl(1)-lat1)+abs(br(2)-lon1);
[min_val,idx]=min(thisTR(:))
[row3,col3]=ind2sub(size(thisTR),idx)
clear thisTR

thisBL = abs(br(1)-lat1)+abs(tl(2)-lon1);
[min_val,idx]=min(thisBL(:))
[row4,col4]=ind2sub(size(thisBL),idx)
clear thisBL


