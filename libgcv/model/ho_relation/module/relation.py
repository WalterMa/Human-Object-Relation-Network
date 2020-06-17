"""Relation Module Definition."""
from __future__ import absolute_import

import math
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class RelationModule(gluon.HybridBlock):
    r"""Relation Module.

    Parameters
    ----------
    num_feat: int, default is 1024
        Dimension number used in fc layers.
    num_group : int, default is 16
        Relation group number.
        dk = num_feat / num_group.
    """
    def __init__(self, num_feat=1024, num_group=16, **kwargs):
        super(RelationModule, self).__init__(**kwargs)
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.fc_position = nn.Dense(self.num_group, activation='relu', weight_initializer=weight_initializer)
            self.fc_q = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.fc_k = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.linear_out = nn.Conv2D(self.num_feat, 1, 1, 0, groups=self.num_group,
                                        weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, feat, ctx_feat, box, ctx_box):
        """Forward Relation Module.

        Parameters
        ----------
        feat : mxnet.nd.NDArray or mxnet.symbol
            (M, 1024) Feature tensor (used to compute q).
        ctx_feat : mxnet.nd.NDArray or mxnet.symbol
            (N, 1024)Contextual Feature tensor (used to compute k,v).
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.

        Returns
        -------
        feat
            (M, 1024).

        """
        pos_embedding = self.position_embedding(F, box, ctx_box, feat_dim=64)  # (M*N, feat_dim)
        pos_feat = self.fc_position(pos_embedding)  # (M*N, num_group)
        pos_feat = pos_feat.transpose()  # (num_group, M*N)

        q_data = self.fc_q(feat)
        q_data = q_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, M, dim_k)
        k_data = self.fc_k(ctx_feat)
        k_data = k_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, N, dim_k)
        v_data = ctx_feat  # (N, 1024)

        qk = F.batch_dot(lhs=q_data, rhs=k_data, transpose_a=False, transpose_b=True)  # (num_group, M, N)
        qk = (1.0 / math.sqrt(float(self.dim_k))) * qk
        pos_feat = F.reshape_like(pos_feat, qk)
        qk = F.transpose(qk, axes=(1, 0, 2))  # (M, num_group, N)
        pos_feat = F.transpose(pos_feat, axes=(1, 0, 2))  # (M, num_group, N)

        weighted_qk = F.log(F.maximum(pos_feat, 1e-6)) + qk
        weighted_qk = F.softmax(data=weighted_qk, axis=2)
        weighted_qk = weighted_qk.reshape((-3, -2))  # (M * num_group, N)

        output = F.dot(lhs=weighted_qk, rhs=v_data)  # (M * num_group, 1024)
        output = output.reshape((-1, self.num_group*self.num_feat, 1, 1))  # (M, num_group*1024, 1, 1)
        output = self.linear_out(output)  # (M, 1024, 1, 1)

        return output.reshape((0, 0))

    def position_embedding(self, F, box, ctx_box, feat_dim=64, wave_length=1000):
        """Compute position embedding.

        Parameters
        ----------
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.
        feat_dim: int, default is 64
        wave_length: int default is 1000

        Returns
        -------
        embedding
            Returns (M, N, feat_dim).
        """
        # position encoding
        # (M, 1)
        xmin, ymin, xmax, ymax = F.split(data=box, num_outputs=4, axis=1)
        box_width = xmax - xmin + 1.
        box_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # (N, 1)
        ctx_xmin, ctx_ymin, ctx_xmax, ctx_ymax = F.split(data=ctx_box, num_outputs=4, axis=1)
        ctx_box_width = ctx_xmax - ctx_xmin + 1.
        ctx_box_height = ctx_ymax - ctx_ymin + 1.
        ctx_center_x = 0.5 * (ctx_xmin + ctx_xmax)
        ctx_center_y = 0.5 * (ctx_ymin + ctx_ymax)

        # (M, N)
        delta_x = F.broadcast_minus(lhs=center_x, rhs=F.transpose(ctx_center_x))
        delta_x = F.broadcast_div(delta_x, box_width)
        delta_x = F.log(F.maximum(F.abs(delta_x), 1e-3))
        delta_y = F.broadcast_minus(lhs=center_y, rhs=F.transpose(ctx_center_y))
        delta_y = F.broadcast_div(delta_y, box_height)
        delta_y = F.log(F.maximum(F.abs(delta_y), 1e-3))
        delta_width = F.broadcast_div(lhs=F.transpose(ctx_box_width), rhs=box_width)
        delta_width = F.log(delta_width)
        delta_height = F.broadcast_div(lhs=F.transpose(ctx_box_height), rhs=box_height)
        delta_height = F.log(delta_height)
        # (M, N, 4)
        position_mat = F.stack(*[delta_x, delta_y, delta_width, delta_height], axis=2)

        # position embedding
        feat_range = F.arange(0, feat_dim / 8)
        dim_mat = F.broadcast_power(lhs=F.full((1,), wave_length), rhs=(8. / feat_dim) * feat_range)
        dim_mat = F.Reshape(dim_mat, shape=(1, 1, 1, -1))  # (1, 1, 1, feat_dim/8)
        # position_mat (M, N, 4, 1)
        position_mat = F.expand_dims(100.0 * position_mat, axis=3)
        div_mat = F.broadcast_div(lhs=position_mat, rhs=dim_mat)  # (M, N, 4, feat_dim/8)
        sin_mat = F.sin(data=div_mat)
        cos_mat = F.cos(data=div_mat)
        embedding = F.concat(sin_mat, cos_mat, dim=3)   # (M, N, 4, feat_dim/4)
        return embedding.reshape((-3, feat_dim))


class HumanObjectRelationModule(gluon.HybridBlock):
    r"""Human-object Relation Module.

    Parameters
    ----------
    num_feat: int, default is 1024
        Dimension number used in fc layers.
    num_group : int, default is 16
        Relation group number.
        dk = num_feat / num_group.
    """
    def __init__(self, num_feat=1024, num_group=16, additional_output=False, **kwargs):
        super(HumanObjectRelationModule, self).__init__(**kwargs)
        self.num_feat = num_feat
        self.num_group = num_group
        self.dim_k = int(num_feat / num_group)
        self.additional_output = additional_output
        weight_initializer = mx.init.Normal(0.01)
        with self.name_scope():
            self.fc_gt_ctx_position = nn.Dense(self.num_group, activation='relu', weight_initializer=weight_initializer)
            self.fc_ctx_gt_position = nn.Dense(self.num_group, activation='relu', weight_initializer=weight_initializer)
            self.fc_gt = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.fc_ctx = nn.Dense(self.num_feat, weight_initializer=weight_initializer)
            self.gt_ctx_linear_out = nn.Conv2D(self.num_feat, 1, 1, 0, groups=self.num_group,
                                               weight_initializer=weight_initializer)
            self.ctx_gt_linear_out = nn.Conv2D(self.num_feat, 1, 1, 0, groups=self.num_group,
                                               weight_initializer=weight_initializer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, feat, ctx_feat, box, ctx_box):
        """Forward Relation Module.

        Parameters
        ----------
        feat : mxnet.nd.NDArray or mxnet.symbol
            (M, 1024) Feature tensor (used to compute q).
        ctx_feat : mxnet.nd.NDArray or mxnet.symbol
            (N, 1024)Contextual Feature tensor (used to compute k,v).
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.

        Returns
        -------
        gt_relation_feat, ctx_relation_feat
            (M, 1024).

        """
        gt_ctx_pos_embedding = self.position_embedding(F, box, ctx_box, feat_dim=64)  # (M*N, feat_dim)
        gt_ctx_pos_feat = self.fc_gt_ctx_position(gt_ctx_pos_embedding)  # (M*N, num_group)
        gt_ctx_pos_feat = gt_ctx_pos_feat.transpose()  # (num_group, M*N)

        ctx_gt_pos_embedding = self.position_embedding(F, ctx_box, box, feat_dim=64)  # (N*M, feat_dim)
        ctx_gt_pos_feat = self.fc_ctx_gt_position(ctx_gt_pos_embedding)  # (N*M, num_group)
        ctx_gt_pos_feat = ctx_gt_pos_feat.transpose()  # (num_group, N*M)

        gt_data = self.fc_gt(feat)
        gt_data = gt_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, M, dim_k)
        ctx_data = self.fc_ctx(ctx_feat)
        ctx_data = ctx_data.reshape((-1, self.num_group, self.dim_k)).transpose(axes=(1, 0, 2))  # (num_group, N, dim_k)

        gt_ctx = F.batch_dot(lhs=gt_data, rhs=ctx_data, transpose_a=False, transpose_b=True)  # (num_group, M, N)
        gt_ctx = (1.0 / math.sqrt(float(self.dim_k))) * gt_ctx
        ctx_gt = F.transpose(gt_ctx, axes=(0, 2, 1))  # (num_group, N, M)

        gt_ctx_pos_feat = F.reshape_like(gt_ctx_pos_feat, gt_ctx)
        gt_ctx = F.transpose(gt_ctx, axes=(1, 0, 2))  # (M, num_group, N)
        gt_ctx_pos_feat = F.transpose(gt_ctx_pos_feat, axes=(1, 0, 2))  # (M, num_group, N)

        weighted_gt_ctx = F.log(F.maximum(gt_ctx_pos_feat, 1e-6)) + gt_ctx
        weighted_gt_ctx = F.softmax(data=weighted_gt_ctx, axis=2)
        weighted_gt_ctx = weighted_gt_ctx.reshape((-3, -2))  # (M * num_group, N)

        gt_output = F.dot(lhs=weighted_gt_ctx, rhs=ctx_feat)  # (M * num_group, 1024)
        gt_output = gt_output.reshape((-1, self.num_group*self.num_feat, 1, 1))  # (M, num_group*1024, 1, 1)
        gt_output = self.gt_ctx_linear_out(gt_output)  # (M, 1024, 1, 1)

        ctx_gt_pos_feat = F.reshape_like(ctx_gt_pos_feat, ctx_gt)  # (num_group, N, M)
        ctx_gt = F.transpose(ctx_gt, axes=(1, 0, 2))  # (N, num_group, M)
        ctx_gt_pos_feat = F.transpose(ctx_gt_pos_feat, axes=(1, 0, 2))  # (N, num_group, M)

        weighted_ctx_gt = F.log(F.maximum(ctx_gt_pos_feat, 1e-6)) + ctx_gt
        weighted_ctx_gt = F.softmax(data=weighted_ctx_gt, axis=2)
        weighted_ctx_gt = weighted_ctx_gt.reshape((-3, -2))  # (N * num_group, M)

        ctx_output = F.dot(lhs=weighted_ctx_gt, rhs=feat)  # (N * num_group, 1024)
        ctx_output = ctx_output.reshape((-1, self.num_group * self.num_feat, 1, 1))  # (N, num_group*1024, 1, 1)
        ctx_output = self.ctx_gt_linear_out(ctx_output)  # (N, 1024, 1, 1)

        if self.additional_output:
            # (M * num_group, N) -> # (M, num_group, N) -> # (M, N)
            gt_ctx_relation = F.mean(weighted_gt_ctx.reshape(-4, -1, self.num_group, -2), axis=1, keepdims=False)
            return gt_output.reshape((0, 0)), ctx_output.reshape((0, 0)), gt_ctx_relation

        return gt_output.reshape((0, 0)), ctx_output.reshape((0, 0))

    def position_embedding(self, F, box, ctx_box, feat_dim=64, wave_length=1000):
        """Compute position embedding.

        Parameters
        ----------
        box: mxnet.nd.NDArray or mxnet.symbol
            (M, 4) boxes with corner encoding.
        ctx_box: mxnet.nd.NDArray or mxnet.symbol
            (N, 4) boxes with corner encoding.
        feat_dim: int, default is 64
        wave_length: int default is 1000

        Returns
        -------
        embedding
            Returns (M, N, feat_dim).
        """
        # position encoding
        # (M, 1)
        xmin, ymin, xmax, ymax = F.split(data=box, num_outputs=4, axis=1)
        box_width = xmax - xmin + 1.
        box_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # (N, 1)
        ctx_xmin, ctx_ymin, ctx_xmax, ctx_ymax = F.split(data=ctx_box, num_outputs=4, axis=1)
        ctx_box_width = ctx_xmax - ctx_xmin + 1.
        ctx_box_height = ctx_ymax - ctx_ymin + 1.
        ctx_center_x = 0.5 * (ctx_xmin + ctx_xmax)
        ctx_center_y = 0.5 * (ctx_ymin + ctx_ymax)

        # (M, N)
        delta_x = F.broadcast_minus(lhs=center_x, rhs=F.transpose(ctx_center_x))
        delta_x = F.broadcast_div(delta_x, box_width)
        delta_x = F.log(F.maximum(F.abs(delta_x), 1e-3))
        delta_y = F.broadcast_minus(lhs=center_y, rhs=F.transpose(ctx_center_y))
        delta_y = F.broadcast_div(delta_y, box_height)
        delta_y = F.log(F.maximum(F.abs(delta_y), 1e-3))
        delta_width = F.broadcast_div(lhs=F.transpose(ctx_box_width), rhs=box_width)
        delta_width = F.log(delta_width)
        delta_height = F.broadcast_div(lhs=F.transpose(ctx_box_height), rhs=box_height)
        delta_height = F.log(delta_height)
        # (M, N, 4)
        position_mat = F.stack(*[delta_x, delta_y, delta_width, delta_height], axis=2)

        # position embedding
        feat_range = F.arange(0, feat_dim / 8)
        dim_mat = F.broadcast_power(lhs=F.full((1,), wave_length), rhs=(8. / feat_dim) * feat_range)
        dim_mat = F.Reshape(dim_mat, shape=(1, 1, 1, -1))  # (1, 1, 1, feat_dim/8)
        # position_mat (M, N, 4, 1)
        position_mat = F.expand_dims(100.0 * position_mat, axis=3)
        div_mat = F.broadcast_div(lhs=position_mat, rhs=dim_mat)  # (M, N, 4, feat_dim/8)
        sin_mat = F.sin(data=div_mat)
        cos_mat = F.cos(data=div_mat)
        embedding = F.concat(sin_mat, cos_mat, dim=3)   # (M, N, 4, feat_dim/4)
        return embedding.reshape((-3, feat_dim))
