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
    jrs = load_jrs(q,qd);
    for t = 1:100
        for j = 1:n_joints
            jrs_mat{j,1}{t,1} = jrs{t,1}{j,1}.Z;
        end
    end
    filename = sprintf('jrs_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'jrs_mat')
end