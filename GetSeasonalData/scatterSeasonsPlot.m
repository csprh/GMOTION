%  P1 = 11587, P2 = 3744, P3 = 3743, P4 = 9306, P5 = 9242, P6 = 3432, P7 = 7816, P8 = 7689
Ps  = [11587, 3744, 3743, 9306, 9242, 3432, 7816, 3689];
Ps1 = Ps+1;


%P1 [-1.3979125, 53.716713]
%P2 [-1.3844931, 53.703594]
%P3 [-1.3845427, 53.70359]
%P4 [-1.3844283, 53.71405]
%P5 [-1.3843958, 53.71393]
%P6 [-1.3830938, 53.703094]
%P7 [-1.3880703, 53.710693]
%P8 [-1.3971751, 53.70243]
load interpLocationNormAC 
arrayAC2 = interpLocation.arrayAC2;

for ii = 1:8
    plot(interpLocation.outcdTSmooth(Ps(ii)+1,:));
    ['P' num2str(ii) '= (Lat, Lon) = (' num2str(interpLocation.lat2(Ps(ii)+1)) ',' num2str(interpLocation.lon2(Ps(ii)+1)) ')']
end
load interpLocationNorm3
interpLocation.arrayAC2 = arrayAC2;

subplot(1,3,1);
plot(interpLocation.arrayAC2,interpLocation.arraySin,'.', 'Color',[0.85,0.33,0.10],'MarkerSize', 1); hold on;
plot(interpLocation.arrayAC2(Ps1(1)),interpLocation.arraySin(Ps1(1)),'+r', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(2)),interpLocation.arraySin(Ps1(2)),'+g', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(3)),interpLocation.arraySin(Ps1(3)),'+b', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(4)),interpLocation.arraySin(Ps1(4)),'+c', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(5)),interpLocation.arraySin(Ps1(5)),'+y', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(6)),interpLocation.arraySin(Ps1(6)),'+k', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(7)),interpLocation.arraySin(Ps1(7)),'*b', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(8)),interpLocation.arraySin(Ps1(8)),'*c', 'MarkerSize', 14);
legend('All Points', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'FontSize', 13); 
title('SIndex$$_{ACF}$$ vs SIndex$$_{Sin}$$','interpreter','latex', 'FontSize', 18);
xlabel('SIndex$$_{ACF}$$','interpreter','latex', 'FontSize', 18);
ylabel('SIndex$$_{Sin}$$','interpreter','latex', 'FontSize', 18);
subplot(1,3,2);
plot(interpLocation.arrayAC2,interpLocation.arrayS,'.', 'Color',[0.85,0.33,0.10], 'MarkerSize', 1); hold on;
plot(interpLocation.arrayAC2(Ps1(1)),interpLocation.arrayS(Ps1(1)),'+r', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(2)),interpLocation.arrayS(Ps1(2)),'+g', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(3)),interpLocation.arrayS(Ps1(3)),'+b', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(4)),interpLocation.arrayS(Ps1(4)),'+c', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(5)),interpLocation.arrayS(Ps1(5)),'+y', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(6)),interpLocation.arrayS(Ps1(6)),'+k', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(7)),interpLocation.arrayS(Ps1(7)),'*b', 'MarkerSize', 14);
plot(interpLocation.arrayAC2(Ps1(8)),interpLocation.arrayS(Ps1(8)),'*c', 'MarkerSize', 14);
title('SIndex$$_{ACF}$$ vs SIndex$$_{STL}$$','interpreter','latex', 'FontSize', 18);
xlabel('SIndex$$_{ACF}$$','interpreter','latex', 'FontSize', 18);
ylabel('SIndex$$_{STL}$$','interpreter','latex', 'FontSize', 18);
subplot(1,3,3);
plot(interpLocation.arrayS,interpLocation.arraySin,'.', 'Color',[0.85,0.33,0.10], 'MarkerSize', 1); hold on;
plot(interpLocation.arrayS(Ps1(1)),interpLocation.arraySin(Ps1(1)),'+r', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(2)),interpLocation.arraySin(Ps1(2)),'+g', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(3)),interpLocation.arraySin(Ps1(3)),'+b', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(4)),interpLocation.arraySin(Ps1(4)),'+c', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(5)),interpLocation.arraySin(Ps1(5)),'+y', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(6)),interpLocation.arraySin(Ps1(6)),'+k', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(7)),interpLocation.arraySin(Ps1(7)),'*b', 'MarkerSize', 14);
plot(interpLocation.arrayS(Ps1(8)),interpLocation.arraySin(Ps1(8)),'*c', 'MarkerSize', 14);
title('SIndex$$_{STL}$$ vs SIndex$$_{Sin}$$','interpreter','latex', 'FontSize', 18);
xlabel('SIndex$$_{STL}$$','interpreter','latex', 'FontSize', 18);
ylabel('SIndex$$_{Sin}$$','interpreter','latex', 'FontSize', 18);
