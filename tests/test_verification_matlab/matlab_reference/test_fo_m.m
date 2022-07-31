clear all
close all

currentFile = mfilename('fullpath');
rootFile = currentFile;
for i = 1:2
rootFile = fileparts(rootFile);
end
cf_path = fullfile(rootFile,'random_config');
cf_key = load(fullfile(cf_path,'config_key.mat'));
N_test = cf_key.N_test; n_joints = cf_key.n_joints;

save_path = fullfile(rootFile,'saved_m');
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

for i = 1:N_test
    cf_filename = sprintf('data_config_%i.mat', i-1);
    cf = load(fullfile(cf_path,cf_filename));
    q = cf.qpos; qd = cf.qvel; qdd = cf.qacc; 
    joint_axes = zeros(3,n_joints); P = joint_axes;
    joint_axes(:) = cf.joint_axes'; P(:) = cf.P';
        
    for j = 1:n_joints
        c = reshape(cf.link_zonos(j,:,1),3,[]);
        G = reshape(cf.link_zonos(j,:,2:end),3,[]);
        link_zonos{j,1} = polyZonotope_ROAHM(c,G);
    end
    [FO_link,~,P_motor] = forward_occupancy(q, qd, joint_axes, P, link_zonos);
    
    FO_link_m = z2Z_mat(FO_link,n_joints);
    P_motor_m = z2Z_mat(P_motor,n_joints);
    filename = sprintf('fo_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'FO_link_m');
    filename = sprintf('pm_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'P_motor_m');
end

%FO_link = pz2z(FO_link,n_joints);
%plot_fo(FO_link,n_joints)

function z = pz2z(pz,n_joints)
    for i = 1:n_joints
        for t = 1:100
            c = pz{i,1}{t,1}.c; G = pz{i,1}{t,1}.G;
            Z = [c G];
            z{i,1}{t,1} = zonotope(Z);
        end
    end

end
function Z_mat = z2Z_mat(z,n_joints)
    for i = 1:n_joints
        for t = 1:100
            c = z{i,1}{t,1}.c; G = z{i,1}{t,1}.G;
            Z = [c G];
            Z_mat{i,1}{t,1} = Z;
        end
    end

end


function plot_fo(FO_link,n_joints)
    for i = 1:n_joints
        for t = 1:100
            if rem(t,10) == 1
                plot(FO_link{i,1}{t,1}); hold on;
              
            end
        end
    end
end