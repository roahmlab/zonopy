clc; clear all; close all

obstacle = zonotope([ 6.7284173965454102, -2.8392839431762695,  6.6047487258911133; 1.0000000000000000,  0.0000000000000000,  0.0000000000000000; 0.0000000000000000,  1.0000000000000000,  0.0000000000000000; 0.0000000000000000,  0.0000000000000000,  1.0000000000000000]');
link = zonotope([ 5.4146862392526751, -4.2595135556619779,  5.6224152980979358;-0.1991221303274556, -0.2871286219819726, -0.1562643119838424; 0.0758859504426452, -0.2147549527408939,  0.2979037307806643;-0.2672713248851009,  0.1065110194231105,  0.1448652802879852]');

% obstacle plot
V = vertices(obstacle);
shp = alphaShape(V(1, :)', V(2, :)', V(3, :)', inf);
p = plot(shp);
p.FaceAlpha = 0.1; p.FaceColor = 'red'; p.EdgeColor = 'red';
hold on;
% link plot
V = vertices(link);
shp = alphaShape(V(1, :)', V(2, :)', V(3, :)', inf);
p = plot(shp);
p.FaceAlpha = 0.1; p.FaceColor = 'blue'; p.EdgeColor = 'blue';
hold on;

[A,b,~] = polytope_PH([link.center-obstacle.center,obstacle.generators,link.generators]);
min(b)<1e-6

FRS = load('FRS.mat');
for t = 1:100
    frs = zonotope(reshape(FRS.fo(t,:,:),[],3)');
    frs_r = reduce(frs,'girard',4);
    V = vertices(frs_r);
    shp = alphaShape(V(1, :)', V(2, :)', V(3, :)', inf);
    p = plot(shp);
    p.FaceAlpha = 0; p.FaceColor = 'green'; p.EdgeAlpha = 0.01;p.EdgeColor = 'green';
    hold on;
    

    [A,b,~] = polytope_PH([frs.center-obstacle.center, frs.generators,obstacle.generators]);
    if min(b)>1e-6
        a=1111
    end
    

end 
%axis([-8,8,-8,8,-8,8])


