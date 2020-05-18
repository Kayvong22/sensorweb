% Locating the Source of Events in Power Dist. Systems Using Micro-PMU Data
% Kayvon Ghahremani

% start from 5 & end with 20

%plot((1:121),Upcc_a_mag)

%-------------Sensor PCC Voltage diff---------
z1 = Upcc_a_mag(5);
theta1 = Upcc_a_angle(5);
[X1,Y1] = pol2cart(theta1,z1);

z2 = Upcc_a_mag(20);
theta2 = Upcc_a_angle(20);
[X2,Y2] = pol2cart(theta2,z2);

%-------------Sensor PCC Current Diff---------
z3 = Upcc_a_mag(5);
theta3 = Upcc_a_angle(5);
[X3,Y3] = pol2cart(theta3,z3);

z4 = Upcc_a_mag(20);
theta4 = Upcc_a_angle(20);
[X4,Y4] = pol2cart(theta4,z4);

%-------------Sensor 701 Voltage Diff---------
z5 = Ua_701_mag(5);
theta5 = Ua_701_angle(5);
[X5,Y5] = pol2cart(theta5,z5);

z6 = Ua_701_mag(20);
theta6 = Ua_701_angle(20);
[X6,Y6] = pol2cart(theta6,z6);

%-------------Sensor 701 Current Diff---------
z7 = Upcc_a_mag(5);
theta7 = Upcc_a_angle(5);
[X7,Y7] = pol2cart(theta7,z7);

z8 = Upcc_a_mag(20);
theta8 = Upcc_a_angle(20);
[X8,Y8] = pol2cart(theta8,z8);

%-------------DeltaPCC Calculations-----------
DV_pcc = ((X2 - X1) + 1i*(Y2 - Y1));
DI_pcc = ((X4 - X3) + 1i*(Y4 - Y3));

%-------------Delta701 Calculations-----------
DV_701 = ((X6 - X5) + 1i*(Y6 - Y5));
DI_701 = ((X8 - X7) + 1i*(Y8 - Y7));

%-------------Impedance, Z Calculations-------
Z_pcc


