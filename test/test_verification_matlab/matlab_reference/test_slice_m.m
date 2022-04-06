clear all
close all

currentFile = mfilename('fullpath');
rootFile = currentFile;
for i = 1:2
rootFile = fileparts(rootFile);
end
test_path = fullfile(rootFile,'test1');

save_path = fullfile(rootFile,'saved_m');
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

N_test = 50;


for i = 1:N_test
    filename = sprintf('slice_test_%i.mat', i-1);
    random_pz = load(fullfile(test_path,filename));

    pz = polyZonotope_ROAHM(random_pz.c,random_pz.G,random_pz.Grest,double(random_pz.expMat),random_pz.id+1);
    pz1 = getSubset(pz,random_pz.slice_i_one+1,random_pz.slice_v_one);
    pz2 = pz;
    for j = 1:2
        pz2 = getSubset(pz2,random_pz.slice_i_two(j)+1,random_pz.slice_v_two(j));
    end
    pz3 = pz;
    for j = 1:3
        pz3 = getSubset(pz3,random_pz.slice_i_three(j)+1,random_pz.slice_v_three(j));
    end
    P={};
    P.c = pz1.c; P.g = pz1.G; P.G = pz1.Grest; P.e = pz1.expMat; P.i = pz1.id-1;
    P.cc = pz2.c; P.gg = pz2.G; P.GG = pz2.Grest; P.ee = pz2.expMat; P.ii = pz2.id-1;
    P.ccc = pz3.c; P.ggg = pz3.G; P.GGG = pz3.Grest; P.eee = pz3.expMat; P.iii = pz3.id-1;
    filename = sprintf('slice_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'P');
end