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

    jrs = load_jrs(q,qd);
    for t = 1:100
        rotato{t,1} = get_transforms_from_jrs(jrs{t,1}, joint_axes);
    end
    rotato_mat = rotato2mat(rotato,n_joints);
  
    filename = sprintf('rotato_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'rotato_mat')
end


function Z_mat = rotato2mat(ro,n_joints)
    for i = 1:n_joints
        for t = 1:100
            G = ro{t,1}{i,1}.G;
            Z = zeros(3,3,size(G,3)+1);
            Z(:,:,1)=ro{t,1}{i,1}.C; Z(:,:,2:end)=G;
            Z_mat{i,1}{t,1} = Z;
        end
    end

end
