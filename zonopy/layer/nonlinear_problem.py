import numpy as np
import torch 
class nlp_setup():
    x_prev = np.zeros(n_links) * np.nan

    def objective(nlp, x):
        qplan = qpos + qvel * T_PLAN + 0.5 * x * T_PLAN ** 2
        return torch.sum(wrap_to_pi(qplan - qgoal) ** 2)

    def gradient(nlp, x):
        qplan = qpos + qvel * T_PLAN + 0.5 * x * T_PLAN ** 2
        return (T_PLAN ** 2 * wrap_to_pi(qplan - qgoal)).numpy()

    def constraints(nlp, x):
        ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
        if (nlp.x_prev != x).any():
            cons_obs = torch.zeros(M)
            grad_cons_obs = torch.zeros(M, n_links)
            # velocity min max constraints
            possible_max_min_q_dot = torch.vstack((qvel, qvel + x * T_PLAN, torch.zeros_like(qvel)))
            q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
            q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
            grad_q_max = torch.diag(T_PLAN * (q_dot_max_idx % 2))
            grad_q_min = torch.diag(T_PLAN * (q_dot_min_idx % 2))
            cons_obs[-2 * n_links:] = torch.hstack((q_dot_max, q_dot_min))
            grad_cons_obs[-2 * n_links:] = torch.vstack((grad_q_max, grad_q_min))
            # velocity min max constraints
            for j in range(n_links):
                c_k = FO_link[j][idx].center_slice_all_dep(ka / g_ka)
                grad_c_k = FO_link[j][idx].grad_center_slice_all_dep(ka / g_ka) / g_ka
                for o in range(n_obs):
                    cons, ind = torch.max((As[j][o][idx] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][idx],
                                            -1)  # shape: n_timsteps, SAFE if >=1e-6
                    grad_cons = (As[j][o][idx].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1,
                                                                                                dimension)) @ grad_c_k).squeeze(
                        -2)  # shape: n_timsteps, n_links safe if >=1e-6
                    cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = cons
                    grad_cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = grad_cons
            nlp.cons_obs = cons_obs.numpy()
            nlp.grad_cons_obs = grad_cons_obs.numpy()
            nlp.x_prev = np.copy(x)
        return nlp.cons_obs

    def jacobian(nlp, x):
        ka = torch.tensor(x, dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps, 1)
        if (nlp.x_prev != x).any():
            cons_obs = torch.zeros(M)
            grad_cons_obs = torch.zeros(M, n_links)
            # velocity min max constraints
            possible_max_min_q_dot = torch.vstack((qvel, qvel + x * T_PLAN, torch.zeros_like(qvel)))
            q_dot_max, q_dot_max_idx = possible_max_min_q_dot.max(0)
            q_dot_min, q_dot_min_idx = possible_max_min_q_dot.min(0)
            grad_q_max = torch.diag(T_PLAN * (q_dot_max_idx % 2))
            grad_q_min = torch.diag(T_PLAN * (q_dot_min_idx % 2))
            cons_obs[-2 * n_links:] = torch.hstack((q_dot_max, q_dot_min))
            grad_cons_obs[-2 * n_links:] = torch.vstack((grad_q_max, grad_q_min))
            # velocity min max constraints
            for j in range(n_links):
                c_k = FO_link[j][idx].center_slice_all_dep(ka / g_ka)
                grad_c_k = FO_link[j][idx].grad_center_slice_all_dep(ka / g_ka) / g_ka
                for o in range(n_obs):
                    cons, ind = torch.max((As[j][o][idx] @ c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][idx],
                                            -1)  # shape: n_timsteps, SAFE if >=1e-6
                    grad_cons = (As[j][o][idx].gather(-2, ind.reshape(n_timesteps, 1, 1).repeat(1, 1,
                                                                                                dimension)) @ grad_c_k).squeeze(
                        -2)  # shape: n_timsteps, n_links safe if >=1e-6
                    cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = cons
                    grad_cons_obs[(j + n_links * o) * n_timesteps:(j + n_links * o + 1) * n_timesteps] = grad_cons
            nlp.cons_obs = cons_obs.numpy()
            nlp.grad_cons_obs = grad_cons_obs.numpy()
            nlp.x_prev = np.copy(x)
        return nlp.grad_cons_obs

    def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                        d_norm, regularization_size, alpha_du, alpha_pr,
                        ls_trials):
        pass