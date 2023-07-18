import torch
from typing import Union
import numpy as np
import math
import zonopy as zp

# Rewrite it in a more basic way to start split
class BaseArmTrajectory:
    def __init__(self,
                 q0:     Union[torch.Tensor, np.ndarray],
                 qd0:    Union[torch.Tensor, np.ndarray],
                 qdd0:   Union[torch.Tensor, np.ndarray],
                 kparam: Union[torch.Tensor, np.ndarray],
                 krange: Union[torch.Tensor, np.ndarray],
                 tbrake: float = 0.5,
                 tfinal: float = 1.0):
        # Save the parameters
        self.q0 = q0
        self.qd0 = qd0
        self.qdd0 = qdd0
        self.kparam = kparam
        self.krange = krange
        self.tbrake = tbrake
        self.tfinal = tfinal

        # Scale kparam from -1 to 1 to the output range
        self._param = 0.5 * (self.kparam + 1) \
                       * (self.krange[1,:] - self.krange[0,:]) + self.krange[0,:]
    
    def getReference(self, times: Union[torch.Tensor, np.ndarray]):
        pass


class PiecewiseArmTrajectory(BaseArmTrajectory):
    def __init__(self,
                 q0:     Union[torch.Tensor, np.ndarray],
                 qd0:    Union[torch.Tensor, np.ndarray],
                 qdd0:   Union[torch.Tensor, np.ndarray],
                 kparam: Union[torch.Tensor, np.ndarray],
                 krange: Union[torch.Tensor, np.ndarray],
                 tbrake: float = 0.5,
                 tfinal: float = 1.0):
        super().__init__(q0, qd0, qdd0, kparam, krange, tbrake, tfinal)

        # Precompute some extra parameters
        stopping_time = (self.tfinal - self.tbrake)

        self._qpeak = self.q0 \
            + self.qd0 * self.tbrake \
            + 0.5 * self._param * self.tbrake * self.tbrake
        self._qdpeak = self.qd0 \
            + self._param * self.tbrake
        
        self._stopping_qdd = (0 - self._qdpeak) * (1.0 / stopping_time)

        self._final_q = self._qpeak \
            + self._qdpeak * stopping_time \
            + 0.5 * self._stopping_qdd * stopping_time * stopping_time

    def getReference(self, times: Union[torch.Tensor, np.ndarray]):
        if isinstance(times, torch.Tensor):
            return self._getReferenceTorchImpl(times)
        elif isinstance(times, np.ndarray):
            return self._getReferenceNpImpl(times)
        elif isinstance(times, zp.batchPolyZonotope):
            return self._getReferenceBatchZpImpl(times)
        else:
            raise TypeError

    def _getReferenceBatchZpImpl(self, times: zp.batchPolyZonotope):
        mask_plan = times.c <= self.tbrake
        mask_stopping = ((times.c < self.tfinal) - mask_plan).to(bool)
        num_t = len(times.c)
        
        raise NotImplementedError
        # q_out_first = None
        # qd_out_first = None
        # qdd_out_first = None

        # # First half of the trajectory
        # if torch.any(mask_plan):
        #     t = times[mask_plan]
        #     q_out[mask_plan,:] = self.q0 \
        #         + torch.outer(t, self.qd0) \
        #         + 0.5 * torch.outer(t*t, self._param)
        #     qd_out[mask_plan,:] = self.qd0 \
        #         + torch.outer(t, self._param)
        #     qdd_out[mask_plan,:] = self._param
        
        # # Second half of the trajectory
        # if torch.any(mask_stopping):
        #     t = times[mask_stopping]
        #     q_out[mask_stopping,:] = self._qpeak \
        #         + torch.outer(t, self._qdpeak) \
        #         + 0.5 * torch.outer(t*t, self._stopping_qdd)
        #     qd_out[mask_stopping,:] = self._qdpeak \
        #         + torch.outer(t, self._stopping_qdd)
        #     qdd_out[mask_stopping,:] = self._stopping_qdd

        # return (q_out, qd_out, qdd_out)

    def _getReferenceNpImpl(self, times: np.ndarray):
        mask_plan = np.zeros_like(times, dtype=bool)
        mask_stopping = np.zeros_like(times, dtype=bool)
        num_t = len(times)

        for i,v in enumerate(times):
            val = v
            if hasattr(v, 'c'):
                val = v.c
            mask_plan[i] = val <= self.tbrake
            mask_stopping[i] = (val < self.tfinal) + ~mask_plan[i]
        
        q_out = np.tile(self._final_q,(num_t,1))
        qd_out = np.tile(self._final_q * 0,(num_t,1))
        qdd_out = np.tile(self._final_q * 0,(num_t,1))
        
        # First half of the trajectory
        if np.any(mask_plan):
            t = times[mask_plan]
            q_out[mask_plan,:] = self.q0 \
                + np.outer(t, self.qd0) \
                + 0.5 * np.outer(t*t, self._param)
            qd_out[mask_plan,:] = self.qd0 \
                + np.outer(t, self._param)
            qdd_out[mask_plan,:] = self._param
        
        # Second half of the trajectory
        if np.any(mask_stopping):
            t = times[mask_stopping]
            q_out[mask_stopping,:] = self._qpeak \
                + np.outer(t, self._qdpeak) \
                + 0.5 * np.outer(t*t, self._stopping_qdd)
            qd_out[mask_stopping,:] = self._qdpeak \
                + np.outer(t, self._stopping_qdd)
            qdd_out[mask_stopping,:] = self._stopping_qdd

        return (q_out, qd_out, qdd_out)

    def _getReferenceTorchImpl(self, times: torch.Tensor):
        mask_plan = times <= self.tbrake
        mask_stopping = ((times < self.tfinal) - mask_plan).to(bool)
        num_t = len(times)
        
        q_out = torch.tile(self._final_q,(num_t,1))
        qd_out = torch.zeros_like(q_out)
        qdd_out = torch.zeros_like(q_out)
        
        # First half of the trajectory
        if torch.any(mask_plan):
            t = times[mask_plan]
            q_out[mask_plan,:] = self.q0 \
                + torch.outer(t, self.qd0) \
                + 0.5 * torch.outer(t*t, self._param)
            qd_out[mask_plan,:] = self.qd0 \
                + torch.outer(t, self._param)
            qdd_out[mask_plan,:] = self._param
        
        # Second half of the trajectory
        if torch.any(mask_stopping):
            t = times[mask_stopping]
            q_out[mask_stopping,:] = self._qpeak \
                + torch.outer(t, self._qdpeak) \
                + 0.5 * torch.outer(t*t, self._stopping_qdd)
            qd_out[mask_stopping,:] = self._qdpeak \
                + torch.outer(t, self._stopping_qdd)
            qdd_out[mask_stopping,:] = self._stopping_qdd

        return (q_out, qd_out, qdd_out)


class BernsteinArmTrajectory(BaseArmTrajectory):
    def __init__(self,
                 q0:     Union[torch.Tensor, np.ndarray],
                 qd0:    Union[torch.Tensor, np.ndarray],
                 qdd0:   Union[torch.Tensor, np.ndarray],
                 kparam: Union[torch.Tensor, np.ndarray],
                 krange: Union[torch.Tensor, np.ndarray],
                 tbrake: float = 0.5,
                 tfinal: float = 1.0):
        super().__init__(q0, qd0, qdd0, kparam, krange, tbrake, tfinal)

        # Precompute some extra parameters
        self._final_q = self._param + q0
        
        betas = self._match_deg5_bernstein_coeff\
            (q0, qd0, qdd0, self._final_q, 0, 0, dtype=object)

        self._alphas = self._bernstein_to_poly(betas)

    @staticmethod
    def _match_deg5_bernstein_coeff(q0,
                                    qd0,
                                    qdd0,
                                    q1,
                                    qd1,
                                    qdd1,
                                    tspan=1,
                                    dtype=float):
        try:
            beta = np.empty(6,dtype=dtype)

            # Position Constraints
            beta[0] = q0
            beta[5] = q1

            # Velocity Constraints
            beta[1] = q0 + (tspan * qd0) * (1.0/5.0)
            beta[4] = q1 - (tspan * qd1) * (1.0/5.0)

            # Acceleration Constraints
            beta[2] = (tspan * tspan * qdd0) * (1.0/20.0) \
                + (2 * tspan * qd0) * (1.0/5.0) + q0
            beta[3] = (tspan * tspan * qdd1) * (1.0/20.0) \
                - (2 * tspan * qd1) * (1.0/5.0) + q1
            
            return beta
        
        except (ValueError, TypeError) as e:
            # If we already redirected, error
            if dtype == object:
                raise e
            # Redirect
            return BernsteinArmTrajectory._match_deg5_bernstein_coeff \
                (q0, qd0, qdd0, q1, qd1, qdd1, tspan, dtype=object)

    @staticmethod
    def _bernstein_to_poly(betas):
        alphas = np.empty_like(betas)
        deg = len(betas)-1
        # All inclusive loops
        for i in range(deg+1):
            alphas[i] = 0.0
            for j in range(i+1):
                alphas[i] = alphas[i] \
                    + (-1.0)**(i-j) \
                    * float(math.comb(deg, i)) \
                    * float(math.comb(i, j)) \
                    * betas[j]
        return alphas

    def getReference(self, times: Union[torch.Tensor, np.ndarray]):
        if isinstance(times, torch.Tensor):
            return self._getReferenceTorchImpl(times)
        elif isinstance(times, np.ndarray):
            return self._getReferenceNpImpl(times)
        else:
            raise TypeError

    def _getReferenceNpImpl(self, times: np.ndarray):
        mask = np.zeros_like(times, dtype=bool)
        num_t = len(times)

        for i,v in enumerate(times):
            val = v
            if hasattr(v, 'c'):
                val = v.c
            mask[i] = val < self.tfinal
        
        q_out = np.tile(self._final_q,(num_t,1))
        qd_out = q_out * 0
        qdd_out = q_out * 0
        
        # Main trajectory part
        if np.any(mask):
            # Scale time
            scale = 1.0/self.tfinal
            t = times[mask] * scale
            q_out[mask,:] = q_out[mask,:] * 0
            for idx, alpha in enumerate(self._alphas):
                q_out[mask,:] = q_out[mask,:] \
                    + np.outer(t**idx, alpha)
                if idx > 0:
                    qd_out[mask,:] = qd_out[mask,:] \
                        + float(idx) * np.outer(t**(idx-1), alpha)
                if idx > 2:
                    qdd_out[mask,:] = qdd_out[mask,:] \
                        + float(idx * (idx-1)) * np.outer(t**(idx-2), alpha)
            qd_out[mask,:] *= scale
            qdd_out[mask,:] *= scale*scale
        
        return (q_out, qd_out, qdd_out)

    def _getReferenceTorchImpl(self, times: torch.Tensor):
        mask = times < self.tfinal
        num_t = len(times)
        
        q_out = torch.tile(self._final_q,(num_t,1))
        qd_out = torch.zeros_like(q_out)
        qdd_out = torch.zeros_like(q_out)
        
        # Main trajectory part
        if torch.any(mask):
            # Scale time
            scale = 1.0/self.tfinal
            t = times[mask] * scale
            q_out[mask,:] = q_out[mask,:] * 0
            for idx, alpha in enumerate(self._alphas):
                q_out[mask,:] = q_out[mask,:] \
                    + torch.outer(t**idx, alpha)
                if idx > 0:
                    qd_out[mask,:] = qd_out[mask,:] \
                        + float(idx) * torch.outer(t**(idx-1), alpha)
                if idx > 2:
                    qdd_out[mask,:] = qdd_out[mask,:] \
                        + float(idx * (idx-1)) * torch.outer(t**(idx-2), alpha)
            qd_out[mask,:] *= scale
            qdd_out[mask,:] *= scale*scale

        return (q_out, qd_out, qdd_out)
