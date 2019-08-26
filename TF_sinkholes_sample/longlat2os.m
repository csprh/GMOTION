function OSREF = longlat2os(longitude,latitude,units,output),
% LONGLAT2OS  Convert an array of latitudes and longitudes 
%             to an array of either 10 figure UK OS grid references
%             or northings and eastings
%
%    OSREF = LONGLAT2OS( LONGITUDE, LATITUDE, UNITS, OUTPUT )
%
%    The constants and method follow the Ordnance Survey's publication
%    "The ellipsoid and the Transverse Mercantor projection", Geodetic
%     information paper No 1, 2/1988 (version 2.2)
%    It is available in pdf format at 
%      http://www.ordsvy.gov.uk/about_us/pdf/gipaper1.pdf
%    Further information is available at
%      http://www.ordsvy.gov.uk/services/gps-co/50000026.pdf
%    and other documents available at the Ordnance Survey's website
%    Information on the National Grid system is available at
%      http://www.ordsvy.gov.uk/downloads/Natgridpdf/20241.pdf
% 
%    OSREF         is a column vector of strings containing the grid
%                  references or a two column array of northings and 
%                  eastings, depending upon the value of OUTPUT 
%    LONGITUDE     is a column vector of longitude values
%    LATITUDE      is a  column vector of latitude values
%    UNITS         [optional] is either 'degrees' or 'radians' and 
%                  specifies the units of the longitude and latitude
%                  values. Default is 'degrees'.
%    OUTPUT        [optional] defines the output format to be either 
%                  'gridref' or 'northeast' for grid reference
%                  or northings and easting respectively.  Default is 'gridref'.  

% Written by Jon Yearsley 8/4/99 for MATLAB V5 and V6
% j.yearsley@abdn.ac.uk
%
% Bugs corrected in NIV and EII (Steve Worrel) 6/1/2000
%
% Checked against GPS data in the Cairngorms 31/3/2000
%
% Here are the conversion factors for the Ordnance Survey's grid
grid = cell([{'SV'} {'SQ'} {'  '} {'  '} {'  '} {'  '} {'NQ'} {'NL'} {'NF'} ...
      {'NA'} {'HV'} {'HQ'} {'HL'}; ...
  {'SW'} {'SR'} {'SM'} {'SG'} {'SB'} {'NW'} {'NR'} {'NM'} {'NG'} ...
      {'NB'} {'HW'} {'HR'} {'HM'}; ...
  {'SX'} {'SS'} {'SN'} {'SH'} {'SC'} {'NX'} {'NS'} {'NN'} {'NH'} {'NC'} ...
      {'HX'} {'HS'} {'HN'}; ...
  {'SY'} {'ST'} {'SO'} {'SJ'} {'SD'} {'NY'} {'NT'} {'NO'} {'NJ'} {'ND'} ...
      {'HY'} {'HT'} {'HO'}; ...
  {'SZ'} {'SU'} {'SP'} {'SK'} {'SE'} {'NZ'} {'NU'} {'NP'} {'NK'} {'NE'} ...
      {'HZ'} {'HU'} {'HP'}; ...
  {'TV'} {'TQ'} {'TL'} {'TF'} {'TA'} {'OV'} {'OQ'} {'OL'} {'OF'} {'OA'} ...
      {'JV'} {'JQ'} {'JL'}; ...
  {'  '} {'TR'} {'TM'} {'TG'} {'TB'} {'OW'} {'  '} {'  '} {'  '} {'  '} ...
     {'  '} {'  '} {'  '}]);

% Here are some useful constants
Fo = 0.9996012717; % Scale factor on the central meridian
a = 6377563.396;   % Earth's semi-major axis in metres 
b = 6356256.910;   % Earth's semi-minor axis in metres 
n = (a-b)/(a+b);
e2 = (a^2-b^2)/a^2; % Eccentricity
Eo = 400000; % Eastings of the true origin
No = -100000; % Northings of the true origin
phi_o = 49.0*pi/180; % Latitude of true origin
lambda_o = -2.0*pi/180; % longitude of true origin

if nargin<2,
  disp( ' *** Error  : Too few input arguments')
  return;
elseif nargin<3,
  units = 'degrees';
  output = 'gridref';
elseif nargin<4,
  output = 'gridref';  
end

if upper(units(1))=='D',
  longitude = longitude*pi/180;
  latitude = latitude*pi/180;
end

% Calculate the arc of a meridian from phi_o to each latitude
arc = arc_of_meridian(phi_o,latitude,Fo,b,n);

% Radius of curvature of a meridian at latitude phi
rho = Fo*a*(1-e2)./(1-e2*sin(latitude).^2).^1.5;
% The radius of curvature at latitude phi
% perpendicular to a meridian
nu = Fo*a./(1-e2*sin(latitude).^2).^0.5;

eta2 = nu./rho-1; % A measure of the difference between rho and nu

P = longitude - lambda_o;  % Difference  of each longitude from longitude of true origin

% Calculate the first few coefficents in the series expansion for 
% Northings and Easting
NI = arc + No;
NII = nu.*sin(latitude).*cos(latitude)/2;
NIII = nu.*sin(latitude).*cos(latitude).^3.*(5-tan(latitude).^2+9*eta2)/24;
NIV = nu.*sin(latitude).*cos(latitude).^5.*(61-58*tan(latitude).^2+tan(latitude).^4)/720;

EI = nu.*cos(latitude);
EII = nu.*cos(latitude).^3.*(nu./rho - tan(latitude).^2)/6;
EIII = nu.*cos(latitude).^5.*(5-18*tan(latitude).^2+tan(latitude).^4+14*eta2- ...
    58*tan(latitude).^2.*eta2)/120;


northings = NI + P.^2.*NII + P.^4.*NIII + P.^6.*NIV;
eastings  = Eo + P.*EI + P.^3.*EII + P.^5.*EIII;

if upper(output(1))=='G',
  % Calculate the UK grid reference
  
  % Reformat the northings and eastings and extract the first two digits
  % from each to find the U.K. grid code (listed in variable grid)
  northings = num2str(round(northings.*10.^5),'%012.0f');
  eastings = num2str(round(eastings.*10.^5),'%012.0f');
  for i=1:length(longitude),
    if abs(str2num(eastings(i,1:2)))>6 | abs(str2num(northings(i,1:2)))>12,
      % If point lies outside the grid set prefix to be blank
      prefix = '  ';
    elseif str2num(eastings(i,1:2))<0 | str2num(northings(i,1:2))<0,
        % If point lies outside the grid set prefix to be blank
        prefix = '  ';
    else
      prefix = grid{str2num(eastings(i,1:2))+1,str2num(northings(i,1:2))+1};
    end
    if prefix(1)==' ',
      OSREF(i,:) = ' Not on National Grid ';
    else
      OSREF(i,:) = [ prefix eastings(i,3:12) northings(i,3:12)];
    end
  end
else
  % Just return the northings and eastings
  OSREF = [northings eastings];
end
return

function M = arc_of_meridian(phi1,phi2,Fo,b,n),
% Function to calculate the developed arc of a meridian from
% phi1 to phi2
%
M = Fo*b * ( (1+n+5*n^2/4 + 5*n^3/4)*(phi2-phi1) ...
        - (3*n+3*n^2+21*n^3/8)*sin(phi2-phi1).*cos(phi2+phi1) ...
	+ (15*n^2/8+15*n^3/8)*sin(2*(phi2-phi1)).*cos(2*(phi2+phi1)) ...
	- 35*n^3/24*sin(3*(phi2-phi1)).*cos(3*(phi2+phi1)) );
