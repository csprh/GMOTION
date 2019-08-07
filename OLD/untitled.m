start = [30000 9000]
start = [30000 9000 1]
count = [1000 1000 1]
ls
p = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
imagesc(p)
start = [30000 19000 1]
p = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
start = [30000 12000 100]
p = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
imagesc(p)
imagesc(log(p))
max(p(:))
min(p(:))
start = [60000 12000 100]
p = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
imagesc(p)
start = [60000 12000 1]
count = [1 1 100]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
figure
plot(p2)
size(p2)
size(p2(:))
plot(p2(:))
start = [61000 12000 1]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
figure
plot(p2(:))
2
p2
count = [5000 5000 100]
count = [5000 5000 1]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
imagesc(p2)
start = [61000 12000 100]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
imagesc(p2)
%-- 27/03/19  8:35:06 am GMT --%
start = [61000 12000 100]
count = [5000 5000 100]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
p2 = h5disp('030A_03647_101313-vel.h5');
h5disp('030A_03647_101313-vel.h5');
start = [61000 16000 100]
start = [61000 16000 1]
count = [61000 16000 1]
start = [1000 1000 100]
p2 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
size(p2)
p1 = p2(1:10:end,1:10:end)
p1 = p2(1:10:end,1:10:end);
imagesc(p1)
imagesc(p1')
p1 = p2(1:3:end,1:3:end);
imagesc(p1')
imagesc(p1)
imagesc(p1')
imagesc(p1)
start = [1000 1000 170]
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
p4 = p3(1:3:end,1:3:end);
imagesc(p4)
%-- 27/03/19  5:04:57 pm GMT --%
ls -alt
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
h5disp('030A_03647_101313-vel.h5');
start = [1 1 170]
count = [68940 18450 1]
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
size(p3)
imagesc(p3)
sum(isnan(p3(:)))
sum(isnan(p3(:)))/(68940*18450)
start = [1 1 171]
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
sum(isnan(p3(:)))/(68940*18450)
figure
imagesc(p3)
start = [1 1 169]
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
figure
imagesc(p3)
l1 = h5read('030A_03647_101313-vel.h5','/Longitude');
imagesc(l1)
lt1 = h5read('030A_03647_101313-vel.h5','/Latitude');
thisInd = isnan(p3);
sum(thisInd(:))
sum(~thisInd(:))
lt1 = h5read('030A_03647_101313-vel.h5','/Latitude',start);
start
count
lt1 = h5read('030A_03647_101313-vel.h5','/Latitude',[1 1],[68941 18482]);
clear
lt1 = h5read('030A_03647_101313-vel.h5','/Latitude',[1 1],[68941 18482]);
ln1 = h5read('030A_03647_101313-vel.h5','/Longitude',[1 1],[68941 18482]);
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',[1 1], [ 68941 18482] );
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',[1 1 100], [ 68941 18482 1] );
lt1 = h5read('030A_03647_101313-vel.h5','/Latitude',[1 1],[68941 18482]/2);
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',[1 1], [ 68941 18482]/2 );
p3 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',[1 1 100], [ 68941/2 18482/2 1] );
ln1 = h5read('030A_03647_101313-vel.h5','/Longitude',[1 1],[68941 18482]/2);
thisInd = isnan(p3);
p4 = p3(~thisInd);
34470*9241
ln2 = ln1(~thisInd);
lt2 = lt1(~thisInd);
plot(lt2,ln2,p4,'+');
plot([lt2 ln2],p4,'+');
plot(lt2,ln2,'+');
scatter(lt2,ln2,p4);
scatter3(lt2,ln2,p4);
%-- 12/04/19  4:07:26 pm BST --%
edit