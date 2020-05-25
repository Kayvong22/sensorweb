% Locating the Source of Events in Power Dist. Systems Using Micro-PMU Data
% Kayvon Ghahremani

%plot((1:121),Ua_701_angle)

%-------------Sensor PCC Voltage diff---------
z1 = Upcc_a_mag(5);
theta1 = Upcc_a_angle(5);
[X1,Y1] = pol2cart(theta1,z1);

z2 = Upcc_a_mag(54);
theta2 = Upcc_a_angle(54);
[X2,Y2] = pol2cart(theta2,z2);

%-------------Sensor PCC Current Diff---------
z3 = Ipcc_a_mag(5);
theta3 = Ipcc_a_angle(5);
[X3,Y3] = pol2cart(theta3,z3);

z4 = Ipcc_a_mag(54);
theta4 = Ipcc_a_angle(54);
[X4,Y4] = pol2cart(theta4,z4);

%-------------Sensor 701 Voltage Diff---------
z5 = Ua_701_mag(5);
theta5 = Ua_701_angle(5);
[X5,Y5] = pol2cart(theta5,z5);

z6 = Ua_701_mag(54);
theta6 = Ua_701_angle(54);
[X6,Y6] = pol2cart(theta6,z6);

%-------------Sensor 701 Current Diff---------
z7 = Ia_701_mag(5);
theta7 = Ia_701_angle(5);
[X7,Y7] = pol2cart(theta7,z7);

z8 = Ia_701_mag(54);
theta8 = Ia_701_angle(54);
[X8,Y8] = pol2cart(theta8,z8);

%-------------DeltaPCC Calculations-----------
DV_pcc = ((X2 - X1) + 1i*(Y2 - Y1));
DI_pcc = ((X4 - X3) + 1i*(Y4 - Y3));

%-------------Delta701 Calculations-----------
DV_701 = ((X6 - X5) + 1i*(Y6 - Y5));
DI_701 = ((X8 - X7) + 1i*(Y8 - Y7));

%-------------Impedance, Z Calculations-------
Z_pcc = (DV_pcc/DI_pcc);
Z_701 = (DV_701/DI_701);
%Take the Real{Z}
Z_pcc_Real = real(Z_pcc);
Z_701_Real = real(Z_701);

Zpcc = ispos(Z_pcc_Real);
Z701 = ispos(Z_701_Real);

if (Zpcc == 1) && (Z701 == 1)
    disp('Real{Zpcc} and Real{Z701} are both positive.')
    disp('The event occurred between the two PMUs.')
elseif (Zpcc == 0)
    disp('Real{Zpcc} is negative.') 
    disp('The event occurred upstream of PMU pcc')
elseif (Z701 == 0)
    disp('Real{Z701} is negative.') 
    disp('The event occurred downstream of PMU 701')
end

function answer = ispos(value)
    if value > 0
        answer = 1;
    else
        answer = 0;
    end
end
