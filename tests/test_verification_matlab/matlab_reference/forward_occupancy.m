function [FO_link,R_motor,P_motor] = forward_occupancy(q, qd, joint_axes, P, link_zonos)

jrs = load_jrs(q, qd);
for t = 1:100
    rotato{t,1} = get_transforms_from_jrs(jrs{t,1}, joint_axes);
end

for t = 1:100
    
    for i = 1:size(q,2)
        P_motor{i,1}{t,1} = polyZonotope_ROAHM(zeros(3,1),[],[]);
        R_motor{i,1}{t,1} = matPolyZonotope_ROAHM(eye(3),[],[]);
        for j = 1:i
            P_motor_j = P(:,j);
            for j_sub = flip(1:j-1)
                P_motor_j =  rotato{t,1}{j_sub,1}*P_motor_j;
            end
            P_motor{i,1}{t,1} = P_motor_j + P_motor{i,1}{t,1};
            %P_motor{i,1}{t,1} = R_motor{i,1}{t,1}*P(:,j) + P_motor{i,1}{t,1};
            %R_motor{i,1}{t,1} = R_motor{i,1}{t,1}*rotato{t,1}{j,1};
        end
        P_link = link_zonos{i,1};
        for i_sub = flip(1:i)
            P_link =  rotato{t,1}{i_sub,1}*P_link;
        end
        FO_link{i,1}{t,1} = P_link+P_motor{i,1}{t,1};
    end
end
end