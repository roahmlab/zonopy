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
    q = cf.qpos; qd = cf.qvel;
    joint_axes = zeros(3,n_joints); P = joint_axes;
    joint_axes(:) = cf.joint_axes'; P(:) = cf.P';
        
    for j = 1:n_joints
        c = reshape(cf.link_zonos(j,:,1),3,[]);
        G = reshape(cf.link_zonos(j,:,2:end),3,[]);
        link_zonos{j,1} = polyZonotope_ROAHM(c,G);
    end

    jrs = load_jrs(q,qd);
    for t = 1:100
        rotato{t,1} = get_transforms_from_jrs(jrs{t,1}, joint_axes);
    end
    for t = 1:100
        r0_r1_l{1,1}{t,1} = rotato{t,1}{1,1}*(rotato{t,1}{2,1}*link_zonos{2,1});
        pm1{1,1}{t,1}  = P(:,1)+rotato{t,1}{1,1}*P(:,2);
        fo1{1,1}{t,1}  = r0_r1_l{1,1}{t,1} + pm1{1,1}{t,1};
    end
    r0_r1_l_mat = z2Z_mat(r0_r1_l,1);
    pm1_mat = z2Z_mat(pm1,1);
    fo1_mat = z2Z_mat(fo1,1);

    filename = sprintf('r0_r1_l_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'r0_r1_l_mat')

    filename = sprintf('pm1_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'pm1_mat')

    filename = sprintf('fo1_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'fo1_mat')
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