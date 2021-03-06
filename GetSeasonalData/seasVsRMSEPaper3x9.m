function seasVsRMSE
clear all
close all
load outputList2;
load interpLocationNormAC 
arrayAC2 = interpLocation.arrayAC2;

load interpLocationNorm3
interpLocation.arrayAC2 = arrayAC2;

arrayAC2 = interpLocation.arrayAC2;
arraySin = interpLocation.arraySin;
arrayS = interpLocation.arrayS;

arrayAC2 = arrayAC2(ouputList2+1);
arraySin = arraySin(ouputList2+1);
arrayS = arrayS(ouputList2+1);


l8In = readNPY('../predictInSAR/RMSEAnal/M-Seq2Seq-LSTMM_R.npy');
l9In = readNPY('../predictInSAR/RMSEAnal/M-Seq2Seq-Sarima_R.npy');
l10In = readNPY('../predictInSAR/RMSEAnal/M-Seq2Seq-Sinu_R.npy');


%l5In9 = l5In(:,9);
%thisInd = l5In9<15;
%l5In9x = l5In9(thisInd);
%arraySx = arrayS(thisInd);
%arraySinx = arraySin(thisInd);
%arrayAC2x = arrayAC2(thisInd);
l8In9 = l8In(:,9);
l9In9 = l9In(:,9);
l10In9 = l10In(:,9);
%save l10In9 l5In9 l8In9 l9In9 l10In9 arrayS arraySin arrayAC2 arraySx arraySinx arrayAC2x l5In9x

load c4
colormap(c4);
%colormap jet

subplot(9,3,2);

thisPlot(arrayAC2, l8In(:,9)', 1, 'LSTM8', {'RMSE'; [num2str(9) ' Months']},[]);set(gca,'xtick',[])
thisPlot(arrayAC2, l9In(:,9)', 2, 'SARIMA',{'RMSE'; [num2str(9) ' Months']}, []);set(gca,'xtick',[])
thisPlot(arrayAC2, l10In(:,9)', 3 , 'SINU', {'RMSE'; [ num2str(9) ' Months']}, []);set(gca,'xtick',[])
ind = 1;
for ii = 8:-1:2
    ind = ind+3;
    ylab = {'RMSE'; [ num2str(ii) ' Months']};
    thisPlot(arrayAC2, l8In(:,ii)', ind, [], ylab,[]);set(gca,'xtick',[]);
    thisPlot(arrayAC2, l9In(:,ii)', ind+1, [],ylab,[]);set(gca,'xtick',[])
    thisPlot(arrayAC2, l10In(:,ii)', ind+2 , [],ylab,[]);set(gca,'xtick',[])
end
ind = ind+3;
ylab = {'RMSE'; [num2str(1) ' Month']};
thisPlot(arrayAC2, l8In(:,1)', ind, [], ylab,'SIndex$$_{ACF}$$');
thisPlot(arrayAC2, l9In(:,1)', ind+1, [],ylab,'SIndex$$_{ACF}$$');
thisPlot(arrayAC2, l10In(:,1)', ind+2 , [],ylab,'SIndex$$_{ACF}$$');

function thisPlot(xx, yy , num, titley, ylab, index1)

subplot(9,3,num);

dscatter(xx,yy,'plottype','contour'); hold on;
dscatter(xx,yy); hold on;
title(titley);
xlabel(index1,'interpreter','latex', 'FontSize', 17);
ylabel(ylab, 'interpreter','latex', 'FontSize', 12);
Fit = polyfit(xx,yy,1); % x = x data, y = y data, 1 = order of the polynomial i.e a straight line 
plot(xx,polyval(Fit,xx))
ylim([0 12]);




%

%plot(arraySx,l5In9x,'.', 'Color',[0.85,0.33,0.10],'MarkerSize', 1); hold on;


function data = readNPY(filename)
% Function to read NPY files into matlab.
% *** Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types.
% See https://github.com/kwikteam/npy-matlab/blob/master/tests/npy.ipynb for
% more.
%

[shape, dataType, fortranOrder, littleEndian, totalHeaderLength, ~] = readNPYheader(filename);

if littleEndian
    fid = fopen(filename, 'r', 'l');
else
    fid = fopen(filename, 'r', 'b');
end

try

    [~] = fread(fid, totalHeaderLength, 'uint8');

    % read the data
    data = fread(fid, prod(shape), [dataType '=>' dataType]);

    if length(shape)>1 && ~fortranOrder
        data = reshape(data, shape(end:-1:1));
        data = permute(data, [length(shape):-1:1]);
    elseif length(shape)>1
        data = reshape(data, shape);
    end

    fclose(fid);

catch me
    fclose(fid);
    rethrow(me);
end

function [arrayShape, dataType, fortranOrder, littleEndian, totalHeaderLength, npyVersion] = readNPYheader(filename)
% function [arrayShape, dataType, fortranOrder, littleEndian, ...
%       totalHeaderLength, npyVersion] = readNPYheader(filename)
%
% parse the header of a .npy file and return all the info contained
% therein.
%
% Based on spec at http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html

fid = fopen(filename);

% verify that the file exists
if (fid == -1)
    if ~isempty(dir(filename))
        error('Permission denied: %s', filename);
    else
        error('File not found: %s', filename);
    end
end

try
    
    dtypesMatlab = {'uint8','uint16','uint32','uint64','int8','int16','int32','int64','single','double', 'logical'};
    dtypesNPY = {'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'b1'};
    
    
    magicString = fread(fid, [1 6], 'uint8=>uint8');
    
    if ~all(magicString == [147,78,85,77,80,89])
        error('readNPY:NotNUMPYFile', 'Error: This file does not appear to be NUMPY format based on the header.');
    end
    
    majorVersion = fread(fid, [1 1], 'uint8=>uint8');
    minorVersion = fread(fid, [1 1], 'uint8=>uint8');
    
    npyVersion = [majorVersion minorVersion];
    
    headerLength = fread(fid, [1 1], 'uint16=>uint16');
    
    totalHeaderLength = 10+headerLength;
    
    arrayFormat = fread(fid, [1 headerLength], 'char=>char');
    
    % to interpret the array format info, we make some fairly strict
    % assumptions about its format...
    
    r = regexp(arrayFormat, '''descr''\s*:\s*''(.*?)''', 'tokens');
    dtNPY = r{1}{1};    
    
    littleEndian = ~strcmp(dtNPY(1), '>');
    
    dataType = dtypesMatlab{strcmp(dtNPY(2:3), dtypesNPY)};
        
    r = regexp(arrayFormat, '''fortran_order''\s*:\s*(\w+)', 'tokens');
    fortranOrder = strcmp(r{1}{1}, 'True');
    
    r = regexp(arrayFormat, '''shape''\s*:\s*\((.*?)\)', 'tokens');
    shapeStr = r{1}{1}; 
    arrayShape = str2num(shapeStr(shapeStr~='L'));

    
    fclose(fid);
    
catch me
    fclose(fid);
    rethrow(me);
end
