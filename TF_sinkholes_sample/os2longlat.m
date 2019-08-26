function longlat = os2longlat(osref),
% OS2LONGLAT  Convert a UK Ordnance Survey grid reference
%             to latitude and longitude
%
%    L = OS2LONGLAT( OSREF )
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
%    OSREF  is a column vector of strings containing the grid references
%             OSREF = [ 'HU436789'; 'OA786504' ]
%           The grid reference can be 4,6,8 or 10 digits depending upon the
%           accuracy, but for any one call of OS2LONGLAT all grid references
%           must have the same number of digits.
%    L      is a two column vector, the first column gives the longitude 
%              and the second latitude (in radians)

% Written by Jon Yearsley 22/4/98 for MATLAB V5 and V6
% Checked against GPS data in the Cairngorms 31/3/2000
%
% j.yearsley@abdn.ac.uk
%
% Here are the conversion factors for the Ordnance Survey's grid
HL = [00 12]; HM = [01 12]; HN = [02 12]; HO = [03 12]; HP = [04 12];
Jl = [05 12];

HQ = [00 11]; HR = [01 11]; HS = [02 11]; HT = [03 11]; HU = [04 11];
JQ = [05 11];

HV = [00 10]; HW = [01 10]; HX = [02 10]; HY = [03 10]; HZ = [04 10];
JV = [05 10];

NA = [00 09]; NB = [01 09]; NC = [02 09]; ND = [03 09]; NE = [04 09];
OA = [05 09];

NF = [00 08]; NG = [01 08]; NH = [02 08]; NJ = [03 08]; NK = [04 08];
OF = [05 08];

NL = [00 07]; NM = [01 07]; NN = [02 07]; NO = [03 07]; NP = [04 07];
OL = [05 07];

NQ = [00 06]; NR = [01 06]; NS = [02 06]; NT = [03 06]; NU = [04 06];
OQ = [05 06];

NW = [01 05]; NX = [02 05]; NY = [03 05]; NZ = [04 05]; OV = [05 05];
OW = [06 05];

SB = [01 04]; SC = [02 04]; SD = [03 04]; SE = [04 04]; TA = [05 04];
TB = [06 04];

SG = [01 03]; SH = [02 03]; SJ = [03 03]; SK = [04 03]; TF = [05 03];
TG = [06 03];

SM = [01 02]; SN = [02 02]; SO = [03 02]; SP = [04 02]; TL = [05 02];
TM = [06 02];

SQ = [00 01]; SR = [01 01]; SS = [02 01]; ST = [03 01]; SU = [04 01];
TQ = [05 01]; TR = [06 01];

SV = [00 00]; SW = [01 00]; SX = [02 00]; SY = [03 00]; SZ = [04 00];
TV = [05 00];

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

% Calculate the scaling, a 4 figure reference is in km (scale = 3), 
% 6 figure is in 100's of metres (scale = 2), 
% 8 figure is in 10's of metres (scale = 1),
% and a 10 figure reference is in metres (scale = 0)
ref_size = length(osref(1,:))/2-1;
scale = 5-ref_size;

% Calculate the eastings and northings of the point
%  and give the result in metres
for i = 1:size(osref,1),
  if exist(osref(i,1:2)),
    eval([ ' ref = ' osref(i,1:2) ';']);
    eval([ ' east(i) = (' num2str(ref(1)) '*10^ref_size + ' ...
	  osref(i,3:2+ref_size) ')*10^scale;' ]);
    eval([ ' north(i) = (' num2str(ref(2)) '*10^ref_size + ' ...
	  osref(i,3+ref_size:2+2*ref_size) ')*10^scale;' ]);
  else
    disp([ '*** Error  : ' osref(i,:) ' is not a valid OS grid reference']);
    north(i) = 0;
    east(i) = 0;
  end
end
Et = east-Eo; % True eastings

% Iterate to find the latitude, phi
arc = zeros(size(east));
phi = phi_o*ones(size(east));
while ~isempty(find(abs(north-No-arc)>0.001)),
  phi = new_phi(arc,north,phi,No,a,Fo);
  arc = arc_of_meridian(phi_o,phi,Fo,b,n);
end

% Radius of curvature of a meridian at latitude phi
rho = Fo*a*(1-e2)./(1-e2*sin(phi).^2).^1.5;
% The radius of curvature at latitude phi
% perpendicular to a meridian
nu = Fo*a./(1-e2*sin(phi).^2).^0.5;

eta2 = nu./rho-1; % A measure of the difference between rho and nu

% Calculate the first few coefficents in the series expansion for 
% longitude and latitude
NI = tan(phi)/2./rho./nu;
NII = NI./12./nu.^2.*(5+3*tan(phi).^2+eta2-9*tan(phi.*eta2).^2);
NIII = NI/360./nu.^4.*(61+90*tan(phi).^2+45*tan(phi).^4);

EI = sec(phi)./nu;
EII = EI/6./nu.^2.*(nu./rho+2*tan(phi).^2);
EIII = EI/120./nu.^4.*(5+28*tan(phi).^2+24*tan(phi).^4);
EIIII = EI/5040./nu.^6.*(61+662*tan(phi).^2+1320*tan(phi).^4+ ...
    720*tan(phi).^6);

% Construct the longitude and latitude from the coefficients
longlat = [(lambda_o + Et.*EI + Et.^3.*EII + ...
      Et.^5.*EIII - Et.^7.*EIIII)' , ...
      (phi - Et.^2.*NI + Et.^4.*NII - Et.^6.*NIII)' ];

return

function M = arc_of_meridian(phi1,phi2,Fo,b,n),
% Function to calculate the developed arc of a meridian from
% phi1 to phi2
%
M = Fo*b * ( (1+n+5*n^2/4 + 5*n^3/4)*(phi2-phi1) ...
        - (3*n+3*n^2+21*n^3/8)*sin(phi2-phi1).*cos(phi2+phi1) ...
	+ (15*n^2/8+15*n^3/8)*sin(2*(phi2-phi1)).*cos(2*(phi2+phi1)) ...
	- 35*n^3/24*sin(3*(phi2-phi1)).*cos(3*(phi2+phi1)) );
   
function phi_prime = new_phi(M,north,phi,No,a,Fo),
% Iteration routine for phi

phi_prime = (north-No-M)/(a*Fo) + phi;

