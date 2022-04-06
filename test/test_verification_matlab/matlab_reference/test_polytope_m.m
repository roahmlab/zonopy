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
    filename = sprintf('polytope_test_%i.mat', i-1);
    random_zonotope = load(fullfile(test_path,filename));
    [a, b, c] = polytope_PH(random_zonotope.two);
    [A, B, C] = polytope_PH(random_zonotope.three);    
    polytopes = {}; polytopes.a=a; polytopes.b=b; polytopes.c=c; polytopes.A=A; polytopes.B=B; polytopes.C=C;
    filename = sprintf('polytope_m_%i.mat', i-1);
    save(fullfile(save_path,filename),'polytopes');
end


