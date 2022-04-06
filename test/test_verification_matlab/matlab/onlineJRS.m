clc
clear all
close all
taylor_degree = 1;
joint_axes = [0; 0; 1];
[Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t, id, id_names] = create_jrs_online(0, pi, joint_axes, taylor_degree);


