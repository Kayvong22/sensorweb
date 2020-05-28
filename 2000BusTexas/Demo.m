run('case_ACTIVSg2000.m')
bus = ans.bus;

    
for ii = (1:2000)
    value = bus(ii,[1]);
    if  (1000 < value) && (value < 2000)
       V_mag1(ii) = bus(ii,[8]);
       V_ang1(ii) = bus(ii,[9]);
       P_real1(ii) = bus(ii,[3]);
       P_reac1(ii) = bus(ii,[4]);
       
       r_power1(ii) = sqrt(((P_real1(ii)).^2)+((P_reac1(ii)).^2));
       theta1(ii) = atand((P_reac1(ii))./(P_real1(ii)));
    
       current_mag1(ii) = (r_power1(ii))./(V_mag1(ii));
       current_ang1(ii) = (theta1(ii))-(V_ang1(ii));
       
       Z_mag1(ii) = (V_mag1(ii))./(current_mag1(ii));
       Z_ang1(ii) = (V_ang1(ii)) - (current_ang1(ii));
       
       [x1(ii),y1(ii)] = pol2cart(Z_ang1(ii),Z_mag1(ii));
       
       x1_final(ii) = ispos(x1(ii));
       
    end
end
for iii = (1:90)
   if (x1_final(iii) == 1) && (x1_final(iii+1) == 1)
       fprintf('Error at %d for buses 1000 - 1091\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value2 = bus(ii,[1]);
    if  (2000 < value2) && (value2 < 3000)
       V_mag2(ii) = bus(ii,[8]);
       V_ang2(ii) = bus(ii,[9]);
       P_real2(ii) = bus(ii,[3]);
       P_reac2(ii) = bus(ii,[4]);
       
       r_power2(ii) = sqrt(((P_real2(ii)).^2)+((P_reac2(ii)).^2));
       theta2(ii) = atand((P_reac2(ii))./(P_real2(ii)));
    
       current_mag2(ii) = (r_power2(ii))./(V_mag2(ii));
       current_ang2(ii) = (theta2(ii))-(V_ang2(ii));
       
       Z_mag2(ii) = (V_mag2(ii))./(current_mag2(ii));
       Z_ang2(ii) = (V_ang2(ii)) - (current_ang2(ii));
       
       [x2(ii),y2(ii)] = pol2cart(Z_ang2(ii),Z_mag2(ii));
       
       x2_final(ii) = ispos(x2(ii));
       
    end
end
for iii = (1:223)
   if (x2_final(iii) == 1) && (x2_final(iii+1) == 1)
       fprintf(2,'Error at %d for buses 2000 - 2133\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value3 = bus(ii,[1]);
    if  (3000 < value3) && (value3 < 4000)
       V_mag3(ii) = bus(ii,[8]);
       V_ang3(ii) = bus(ii,[9]);
       P_real3(ii) = bus(ii,[3]);
       P_reac3(ii) = bus(ii,[4]);
       
       r_power3(ii) = sqrt(((P_real3(ii)).^2)+((P_reac3(ii)).^2));
       theta3(ii) = atand((P_reac3(ii))./(P_real3(ii)));
    
       current_mag3(ii) = (r_power3(ii))./(V_mag3(ii));
       current_ang3(ii) = (theta3(ii))-(V_ang3(ii));
       
       Z_mag3(ii) = (V_mag3(ii))./(current_mag3(ii));
       Z_ang3(ii) = (V_ang3(ii)) - (current_ang3(ii));
       
       [x3(ii),y3(ii)] = pol2cart(Z_ang3(ii),Z_mag3(ii));
       
       x3_final(ii) = ispos(x3(ii));
       
      
    end
end
for iii = (1:370)
   if (x3_final(iii) == 1) && (x3_final(iii+1) == 1)
       fprintf('Error at %d for buses 3000 - 3147\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value4 = bus(ii,[1]);
    if  (4000 < value4) && (value4 < 5000)
       V_mag4(ii) = bus(ii,[8]);
       V_ang4(ii) = bus(ii,[9]);
       P_real4(ii) = bus(ii,[3]);
       P_reac4(ii) = bus(ii,[4]);
       
       r_power4(ii) = sqrt(((P_real4(ii)).^2)+((P_reac4(ii)).^2));
       theta4(ii) = atand((P_reac4(ii))./(P_real4(ii)));
    
       current_mag4(ii) = (r_power4(ii))./(V_mag4(ii));
       current_ang4(ii) = (theta4(ii))-(V_ang4(ii));
       
       Z_mag4(ii) = (V_mag4(ii))./(current_mag4(ii));
       Z_ang4(ii) = (V_ang4(ii)) - (current_ang4(ii));
       
       [x4(ii),y4(ii)] = pol2cart(Z_ang4(ii),Z_mag4(ii));
       
       x4_final(ii) = ispos(x4(ii));
       
      
    end
end
for iii = (1:566)
   if (x4_final(iii) == 1) && (x4_final(iii+1) == 1)
       fprintf(2,'Error at %d for buses 4000 - 4197\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value5 = bus(ii,[1]);
    if  (5000 < value5) && (value5 < 6000)
       V_mag5(ii) = bus(ii,[8]);
       V_ang5(ii) = bus(ii,[9]);
       P_real5(ii) = bus(ii,[3]);
       P_reac5(ii) = bus(ii,[4]);
       
       r_power5(ii) = sqrt(((P_real5(ii)).^2)+((P_reac5(ii)).^2));
       theta5(ii) = atand((P_reac5(ii))./(P_real5(ii)));
    
       current_mag5(ii) = (r_power5(ii))./(V_mag5(ii));
       current_ang5(ii) = (theta5(ii))-(V_ang5(ii));
       
       Z_mag5(ii) = (V_mag5(ii))./(current_mag5(ii));
       Z_ang5(ii) = (V_ang5(ii)) - (current_ang5(ii));
       
       [x5(ii),y5(ii)] = pol2cart(Z_ang5(ii),Z_mag5(ii));
       
       x5_final(ii) = ispos(x5(ii));
       
       
    end
end
for iii = (1:1049)
   if (x5_final(iii) == 1) && (x5_final(iii+1) == 1)
       fprintf('Error at %d for buses 5000 - 5485\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value6 = bus(ii,[1]);
    if  (6000 < value6) && (value6 < 7000)
       V_mag6(ii) = bus(ii,[8]);
       V_ang6(ii) = bus(ii,[9]);
       P_real6(ii) = bus(ii,[3]);
       P_reac6(ii) = bus(ii,[4]);
       
       r_power6(ii) = sqrt(((P_real6(ii)).^2)+((P_reac6(ii)).^2));
       theta6(ii) = atand((P_reac6(ii))./(P_real6(ii)));
    
       current_mag6(ii) = (r_power6(ii))./(V_mag6(ii));
       current_ang6(ii) = (theta6(ii))-(V_ang6(ii));
       
       Z_mag6(ii) = (V_mag6(ii))./(current_mag6(ii));
       Z_ang6(ii) = (V_ang6(ii)) - (current_ang6(ii));
       
       [x6(ii),y6(ii)] = pol2cart(Z_ang6(ii),Z_mag6(ii));
       
       x6_final(ii) = ispos(x6(ii));
       
      
    end
end
for iii = (1:1407)
   if (x6_final(iii) == 1) && (x6_final(iii+1) == 1)
       fprintf(2,'Error at %d for buses 6000 - 6360\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value7 = bus(ii,[1]);
    if  (7000 < value7) && (value7 < 8000)
       V_mag7(ii) = bus(ii,[8]);
       V_ang7(ii) = bus(ii,[9]);
       P_real7(ii) = bus(ii,[3]);
       P_reac7(ii) = bus(ii,[4]);
       
       r_power7(ii) = sqrt(((P_real7(ii)).^2)+((P_reac7(ii)).^2));
       theta7(ii) = atand((P_reac7(ii))./(P_real7(ii)));
    
       current_mag7(ii) = (r_power7(ii))./(V_mag7(ii));
       current_ang7(ii) = (theta7(ii))-(V_ang7(ii));
       
       Z_mag7(ii) = (V_mag7(ii))./(current_mag7(ii));
       Z_ang7(ii) = (V_ang7(ii)) - (current_ang7(ii));
       
       [x7(ii),y7(ii)] = pol2cart(Z_ang7(ii),Z_mag7(ii));
       
       x7_final(ii) = ispos(x7(ii));
       
      
    end
end
for iii = (1:1839)
   if (x7_final(iii) == 1) && (x7_final(iii+1) == 1)
       fprintf('Error at %d for buses 7000 - 7432\n',bus(iii,[1]));
   end
end
for ii = (1:2000)
    value8 = bus(ii,[1]);
    if  (8000 < value8) && (value8 < 9000)
       V_mag8(ii) = bus(ii,[8]);
       V_ang8(ii) = bus(ii,[9]);
       P_real8(ii) = bus(ii,[3]);
       P_reac8(ii) = bus(ii,[4]);
       
       r_power8(ii) = sqrt(((P_real8(ii)).^2)+((P_reac8(ii)).^2));
       theta8(ii) = atand((P_reac8(ii))./(P_real8(ii)));
    
       current_mag8(ii) = (r_power8(ii))./(V_mag8(ii));
       current_ang8(ii) = (theta8(ii))-(V_ang8(ii));
       
       Z_mag8(ii) = (V_mag8(ii))./(current_mag8(ii));
       Z_ang8(ii) = (V_ang8(ii)) - (current_ang8(ii));
       
       [x8(ii),y8(ii)] = pol2cart(Z_ang8(ii),Z_mag8(ii));
       
       x8_final(ii) = ispos(x8(ii));
       
  
    end
end
for iii = (1:1999)
   if (x8_final(iii) == 1) && (x8_final(iii+1) == 1)
       fprintf(2,'Error at %d for buses 8000 - 8160\n',bus(iii,[1]));
   end
end


function answer = ispos(value)
    if value > 0
        answer = 1;
    else
        answer = 0;
    end
end
