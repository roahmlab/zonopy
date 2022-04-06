clear all
close all

currentFile = mfilename('fullpath');
rootFile = fileparts(currentFile);

mesh_path = fullfile(rootFile,'assets','robots','fetch_arm','fetch_description','meshes');

TR = stlread(fullfile(mesh_path,'base_link_collision.stl'));
trimesh(TR);

x = TR.Points(:,1); y = TR.Points(:,2); z = TR.Points(:,3);
x_M = max(x); x_m = min(x); y_M = max(y); y_m = min(y); z_M = max(z); z_m = min(z); 

c = [(x_M+x_m)/2, (y_M+y_m)/2, (z_M+z_m)/2];
g = diag([(x_M-x_m)/2, (y_M-y_m)/2, (z_M-z_m)/2]);
