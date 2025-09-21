import torch
# from lie.lie_util import *
from torch import nn


import torch

HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5


def allclose(mat1, mat2, tol=1e-6):
    '''
    check is all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar
    '''
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """
    check element-wise if two tensors are close within some tolerance.

    Either tensot can be replaced by a scalar
    """
    return (mat1 - mat2).abs_().lt(tol)

def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()


def trace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)

    return tr.view(mat.shape[0])


def matR_log_map(R, eps: float = 1e-4, cos_angle: bool = False):
    """
    Returns the axis angle aka lie algebra parameters from a rotation matrix(SO3)
    """
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = trace(R)

    if((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError(
            "A matrix has trace outside valid range [-1-eps, 3+eps]."
        )

    # clamp to valid range
    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)

    # phi ... rotation angle
    phi = 0.5 * (rot_trace - 1.0)
    phi = phi.acos()

    phi_valid = torch.clamp(phi.abs(), eps) * phi.sign()

    log_rot_hat = (phi_valid / (2.0 * phi_valid.sin()))[:, None, None] * (
        R - R.permute(0, 2, 1)
    )
    log_rot = hat_inv(log_rot_hat)
    return log_rot


def lie_u_v(u, v, eps: float = 1e-4):
    """
    find the axis angle parameters to rotate unit vector u onto unit vector v
    which is also the lie algebra parameters of corresponding SO3
    """
    w = torch.cross(u, v, dim=1)

    w_norm = torch.norm(w, p=2, dim=1)
    w_norm = torch.clamp(w_norm, eps)
    A = w / w_norm[:, None] * torch.mul(u, v).sum(dim=1).acos()[:, None]
    return A


def lie_exp_map(log_rot, eps: float = 1e-4):
    """
    Convert the lie algebra parameters to rotation matrices
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * torch.bmm(skews, skews)
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )
    # print(R.shape)
    return R


def hat(v):
    """
    compute the skew-symmetric matrices with a batch of 3d vectors.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    h = v.new_zeros(N, 3, 3)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def hat_inv(h):
    """
    compute the 3d-vectors with a batch of 3x3 skew-symmetric matrics.
    """
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v
############################################################################################################

class LieSkeleton(object):
    def __init__(self, raw_translation, kinematic_tree, tensor):
        super(LieSkeleton, self).__init__()
        self.tensor = tensor
        # print(self.tensor)
        self._raw_translation = self.tensor(raw_translation.shape).copy_(raw_translation).detach()
        self._kinematic_tree = kinematic_tree
        self._translation = None
        self._parents = [0] * len(self._raw_translation)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_translation)

    def raw_translation(self):
        return self._raw_translation

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._translation

    def get_translation_joints(self, joints):
        # joints/offsets (batch_size, joints_num, 3)
        # print(self._raw_translation.shape)
        _translation = self._raw_translation.clone().detach()
        _translation = _translation.expand(joints.shape[0], -1, -1).clone()
        #print(_translation.shape)
        #print(self._raw_translation.shape)
        for i in range(1, self._raw_translation.shape[0]):
            _translation[:, i, :] = torch.norm(joints[:, i, :] - joints[:, self._parents[i], :], p=2, dim=1)[:, None] * \
                                     _translation[:, i, :]
        self._translation = _translation
        return _translation

    def get_translation_bone(self, bonelengths):
        # bonelength (batch_size, joints_num - 1)
        # offsets (batch_size, joints_num, 3)
        self._translation = self._raw_translation.clone().detach().expand(bonelengths.size(0), -1, -1).clone().to(bonelengths.device)
        self._translation[:, 1:, :] = bonelengths * self._translation[:, 1:, :]

    def inverse_kinemetics(self, joints):
        # joints (batch_size, joints_num, 3)
        # lie_params (batch_size, joints_num, 3)
        lie_params = self.tensor(joints.shape).fill_(0)
        # root_matR (batch_size, 3, 3)
        root_matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
        for chain in self._kinematic_tree:
            R = root_matR
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_translation[chain[j + 1]].expand(joints.shape[0], -1).clone().detach().to(joints.device)
                # (batch, 3)
                v = joints[:, chain[j+1], :] - joints[:, chain[j], :]
                # (batch, 3)
                v = v / torch.norm(v, p=2, dim=1)[:, None]
                # (batch, 3, 3)
                R_local = torch.matmul(R.transpose(1, 2), lie_exp_map(lie_u_v(u, v)))
                # print("R_local shape:" + str(R_local.shape))
                # print(R_local)
                # t = lie_u_v(u, v)
                lie_params[:, chain[j + 1], :] = matR_log_map(R_local)
                
                R = torch.matmul(R, R_local)
                
        return lie_params

    def forward_kinematics(self, lie_params, joints, root_translation, do_root_R = False, scale_inds=None):
        # lie_params (batch_size, joints_num, 3) lie_params[:, 0, :] is not used
        # joints (batch_size, joints_num, 3)
        # root_translation (batch_size, 3)
        # translation_mat (batch_size, joints_num, 3)
        translation_mat = self.get_translation_joints(joints)
        if scale_inds is not None:
            translation_mat[:, scale_inds, :] *= 1.25
        joints = self.tensor(lie_params.size()).fill_(0)
        joints[:, 0] = root_translation
        for chain in self._kinematic_tree:
            # if do_root_R is true, root coordinate system has rotation angulers
            # Plus, for chain not containing root(e.g arms), we use root rotation as the rotation
            # of joints near neck(i.e. beginning of this chain).
            if do_root_R:
                matR = lie_exp_map(lie_params[:, 0, :])
            # Or, root rotation matrix is identity matrix, which means no rotation at global coordinate system
            else:
                matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, lie_exp_map(lie_params[:, chain[i], :]))
                translation_vec = translation_mat[:, chain[i], :].unsqueeze_(-1)
                joints[:, chain[i], :] = torch.matmul(matR, translation_vec).squeeze_()\
                                         + joints[:, chain[i-1], :]
        return joints