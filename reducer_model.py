import math
import os
import sys
import time
import numpy as np
import scipy.sparse as ssp
from typing import List, Tuple
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
from pypower.ext2int import ext2int
import cplex
import gurobipy as gp
from gurobipy import GRB
import mosek.fusion as msk
import mosek.fusion.pythonic

hurricane = "Fiona"
# hurricane = "Maria"
# hurricane = "Ernesto"
alpha_obj = 0.05
nt = 24
nd = 956


class Presolver(object):
    def __init__(self):
        print('Start Building Model.')

        self.model_mosek = msk.Model()

        self.model_mosek._t_pos = self.model_mosek.variable(f"t_pos", [nt], msk.Domain.greaterThan(0.))
        self.model_mosek._z_pos = self.model_mosek.variable(f"z_pos", [nt, nd], msk.Domain.unbounded())
        self.model_mosek._v1_pos = self.model_mosek.variable(f"v1_pos", [nt, nd], msk.Domain.greaterThan(0.))
        self.model_mosek._v2_pos = self.model_mosek.variable(f"v2_pos", [nt, nd], msk.Domain.greaterThan(0.))

        self.model_mosek._t_neg = self.model_mosek.variable(f"t_neg", [nt], msk.Domain.greaterThan(0.))
        self.model_mosek._z_neg = self.model_mosek.variable(f"z_neg", [nt, nd], msk.Domain.unbounded())
        self.model_mosek._v1_neg = self.model_mosek.variable(f"v1_neg", [nt, nd], msk.Domain.greaterThan(0.))
        self.model_mosek._v2_neg = self.model_mosek.variable(f"v2_neg", [nt, nd], msk.Domain.greaterThan(0.))

        for t in range(nt):
            self.model_mosek.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model_mosek._v1_pos[t, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_pos[t], nd, 0), nd, 1),
                    - Pd_survive_mean[t].reshape(nd, 1) - msk.Expr.reshape(self.model_mosek._z_pos[t, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            )
            self.model_mosek.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model_mosek._v2_pos[t, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_pos[t], nd, 0), nd, 1),
                    Pd_loss_mean[t].reshape(nd, 1) - msk.Expr.reshape(self.model_mosek._z_pos[t, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            )
            self.model_mosek.constraint(
                msk.Expr.mulElm(failure_prob[t, :], msk.Expr.reshape(self.model_mosek._v1_pos[t, :], nd))
                + msk.Expr.mulElm(1. - failure_prob[t, :], msk.Expr.reshape(self.model_mosek._v2_pos[t, :], nd))
                <= msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_pos[t], nd, 0), nd)
            )

            self.model_mosek.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model_mosek._v1_neg[t, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_neg[t], nd, 0), nd, 1),
                    Pd_survive_mean[t].reshape(nd, 1) - msk.Expr.reshape(self.model_mosek._z_neg[t, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            )
            self.model_mosek.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model_mosek._v2_neg[t, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_neg[t], nd, 0), nd, 1),
                    - Pd_loss_mean[t].reshape(nd, 1) - msk.Expr.reshape(self.model_mosek._z_neg[t, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            )
            self.model_mosek.constraint(
                msk.Expr.mulElm(failure_prob[t, :], msk.Expr.reshape(self.model_mosek._v1_neg[t, :], nd))
                + msk.Expr.mulElm(1. - failure_prob[t, :], msk.Expr.reshape(self.model_mosek._v2_neg[t, :], nd))
                <= msk.Expr.reshape(msk.Expr.repeat(self.model_mosek._t_neg[t], nd, 0), nd)
            )

        self.model_mosek.objective(
            "obj", msk.ObjectiveSense.Minimize,
            msk.Expr.sum(msk.Expr.vstack([
                msk.Expr.sum(self.model_mosek._z_pos[t, :]) - math.log(alpha_obj) * self.model_mosek._t_pos[t]
                + msk.Expr.sum(self.model_mosek._z_neg[t, :]) - math.log(alpha_obj) * self.model_mosek._t_neg[t]
                for t in range(nt)
            ]))
        )
        self.model_mosek.setLogHandler(sys.stdout)
        self.model_mosek.setSolverParam("intpntSolveForm", "dual")
        self.model_mosek.solve()

        z_pos_sol = self.model_mosek._z_pos.level().reshape(nt, nd)
        t_pos_sol = self.model_mosek._t_pos.level()
        z_neg_sol = self.model_mosek._z_neg.level().reshape(nt, nd)
        t_neg_sol = self.model_mosek._t_neg.level()

        self.evar_gen_pos_coeff = np.zeros(nt)
        self.evar_gen_neg_coeff = np.zeros(nt)
        for t in range(nt):
            self.evar_gen_pos_coeff[t] = z_pos_sol[t].sum() - math.log(alpha_obj) * t_pos_sol[t]
            self.evar_gen_neg_coeff[t] = z_neg_sol[t].sum() - math.log(alpha_obj) * t_neg_sol[t]

        self.var_names_to_indices = {}
        self.var_indices_to_names = {}
        self.cons_names_to_indices = {}
        self.cons_indices_to_names = {}

        self.model_cplex = cplex.Cplex()
        self.model_cplex.objective.set_sense(self.model_cplex.objective.sense.minimize)
        for t in range(nt):
            ug_thermal_names = [f'ug_thermal_{t}_{g}' for g in range(ng_thermal)]
            ug_thermal_indices = self.model_cplex.variables.add(
                obj=gencost_thermal[:, 5] * Pg_lb_thermal + gencost_thermal[:, 6],
                types=[self.model_cplex.variables.type.binary] * ng_thermal, names=ug_thermal_names)
            for (name, index) in zip(ug_thermal_names, ug_thermal_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            vg_thermal_names = [f'vg_thermal_{t}_{g}' for g in range(ng_thermal)]
            vg_thermal_indices = self.model_cplex.variables.add(
                obj=gencost_thermal[:, STARTUP],
                types=[self.model_cplex.variables.type.binary] * ng_thermal, names=vg_thermal_names)
            for (name, index) in zip(vg_thermal_names, vg_thermal_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            wg_thermal_names = [f'wg_thermal_{t}_{g}' for g in range(ng_thermal)]
            wg_thermal_indices = self.model_cplex.variables.add(
                obj=np.zeros(ng_thermal),
                types=[self.model_cplex.variables.type.binary] * ng_thermal, names=wg_thermal_names)
            for (name, index) in zip(wg_thermal_names, wg_thermal_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            Pg_gt_lb_thermal_names = [f'Pg_gt_lb_thermal_{t}_{g}' for g in range(ng_thermal)]
            Pg_gt_lb_thermal_indices = self.model_cplex.variables.add(
                obj=gencost_thermal[:, 5],
                types=[self.model_cplex.variables.type.continuous] * ng_thermal, names=Pg_gt_lb_thermal_names)
            for (name, index) in zip(Pg_gt_lb_thermal_names, Pg_gt_lb_thermal_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            betag_thermal_names = [f'betag_thermal_{t}_{g}' for g in range(ng_thermal)]
            betag_thermal_indices = self.model_cplex.variables.add(
                obj=np.zeros(ng_thermal),
                types=[self.model_cplex.variables.type.continuous] * ng_thermal, names=betag_thermal_names)
            for (name, index) in zip(betag_thermal_names, betag_thermal_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name

        for t in range(nt):
            Pg_solar_names = [f'Pg_solar_{t}_{g}' for g in range(ng_solar)]
            Pg_solar_indices = self.model_cplex.variables.add(
                obj=gencost_solar[:, 5],
                lb=Pg_lb_solar[t], ub=Pg_ub_solar[t],
                types=[self.model_cplex.variables.type.continuous] * ng_solar, names=Pg_solar_names)
            for (name, index) in zip(Pg_solar_names, Pg_solar_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            betag_solar_names = [f'betag_solar_{t}_{g}' for g in range(ng_solar)]
            betag_solar_indices = self.model_cplex.variables.add(
                obj=np.zeros(ng_solar),
                types=[self.model_cplex.variables.type.continuous] * ng_solar, names=betag_solar_names)
            for (name, index) in zip(betag_solar_names, betag_solar_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name

        for t in range(nt):
            Pg_hydro_names = [f'Pg_hydro_{t}_{g}' for g in range(ng_hydro)]
            Pg_hydro_indices = self.model_cplex.variables.add(
                obj=gencost_hydro[:, 5], lb=Pg_lb_hydro, ub=Pg_ub_hydro,
                types=[self.model_cplex.variables.type.continuous] * ng_hydro, names=Pg_hydro_names)
            for (name, index) in zip(Pg_hydro_names, Pg_hydro_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name
        for t in range(nt):
            betag_hydro_names = [f'betag_hydro_{t}_{g}' for g in range(ng_hydro)]
            betag_hydro_indices = self.model_cplex.variables.add(
                obj=np.zeros(ng_hydro),
                types=[self.model_cplex.variables.type.continuous] * ng_hydro, names=betag_hydro_names)
            for (name, index) in zip(betag_hydro_names, betag_hydro_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name

        for t in range(nt):
            Fl_names = [f'Fl_{t}_{l}' for l in range(nl)]
            Fl_indices = self.model_cplex.variables.add(
                obj=np.zeros(nl), lb=- Fl_ub, ub=Fl_ub,
                types=[self.model_cplex.variables.type.continuous] * nl, names=Fl_names)
            for (name, index) in zip(Fl_names, Fl_indices):
                self.var_names_to_indices[name] = index
                self.var_indices_to_names[index] = name

        # constraints on startup and shutdown
        for t in range(1, nt):
            for g in range(ng_thermal):
                cons_logical_thermal_names = [f"cons_logical_thermal_{t}_{g}"]
                cons_logical_thermal_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"ug_thermal_{t}_{g}", f"ug_thermal_{t - 1}_{g}", f"vg_thermal_{t}_{g}",
                             f"wg_thermal_{t}_{g}"],
                        val=[1., -1., -1., 1.])],
                    senses=['E'], rhs=[0.], names=cons_logical_thermal_names
                )
                for (name, index) in zip(cons_logical_thermal_names, cons_logical_thermal_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for g in range(ng_thermal):
            for t in range(UT_gen_thermal[g] - 1, nt):
                cons_min_uptime_names = [f"cons_min_uptime_thermal_{t}_{g}"]
                cons_min_uptime_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"vg_thermal_{i}_{g}" for i in range(t - UT_gen_thermal[g] + 1, t + 1)] + [
                            f"ug_thermal_{t}_{g}"],
                        val=[1. for i in range(t - UT_gen_thermal[g] + 1, t + 1)] + [-1.])],
                    senses=['L'], rhs=[0.], names=cons_min_uptime_names
                )
                for (name, index) in zip(cons_min_uptime_names, cons_min_uptime_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for g in range(ng_thermal):
            for t in range(DT_gen_thermal[g] - 1, nt):
                cons_min_downtime_names = [f"cons_min_downtime_thermal_{t}_{g}"]
                cons_min_downtime_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"wg_thermal_{i}_{g}" for i in range(t - DT_gen_thermal[g] + 1, t + 1)] + [
                            f"ug_thermal_{t}_{g}"],
                        val=[1. for i in range(t - DT_gen_thermal[g] + 1, t + 1)] + [1.])],
                    senses=['L'], rhs=[1.], names=cons_min_downtime_names
                )
                for (name, index) in zip(cons_min_downtime_names, cons_min_downtime_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name

        # constraints on ramping
        for g in range(ng_thermal):
            if RUg_thermal[g] < Pg_ub_thermal[g] - Pg_lb_thermal[g]:
                for t in range(1, nt):
                    cons_ramp_up_names = [f"cons_ramp_up_thermal_{t}_{g}"]
                    cons_ramp_up_indices = self.model_cplex.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[f"Pg_gt_lb_thermal_{t}_{g}", f"Pg_gt_lb_thermal_{t - 1}_{g}",
                                 f"ug_thermal_{t}_{g}", f"vg_thermal_{t}_{g}"],
                            val=[1., -1., - RUg_thermal[g], - (Pg_ub_thermal[g] - Pg_lb_thermal[g] - RUg_thermal[g])])],
                        senses=['L'], rhs=[0.], names=cons_ramp_up_names
                    )
                    for (name, index) in zip(cons_ramp_up_names, cons_ramp_up_indices):
                        self.cons_names_to_indices[name] = index
                        self.cons_indices_to_names[index] = name
        for g in range(ng_thermal):
            if RDg_thermal[g] < Pg_ub_thermal[g] - Pg_lb_thermal[g]:
                for t in range(1, nt):
                    cons_ramp_down_names = [f"cons_ramp_down_thermal_{t}_{g}"]
                    cons_ramp_down_indices = self.model_cplex.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[f"Pg_gt_lb_thermal_{t - 1}_{g}", f"Pg_gt_lb_thermal_{t}_{g}",
                                 f"ug_thermal_{t - 1}_{g}", f"wg_thermal_{t}_{g}"],
                            val=[1., -1., - RDg_thermal[g], - (Pg_ub_thermal[g] - Pg_lb_thermal[g] - RDg_thermal[g])])],
                        senses=['L'], rhs=[0.], names=cons_ramp_down_names
                    )
                    for (name, index) in zip(cons_ramp_down_names, cons_ramp_down_indices):
                        self.cons_names_to_indices[name] = index
                        self.cons_indices_to_names[index] = name
        for g in range(ng_hydro):
            if RUg_hydro[g] < Pg_ub_hydro[g] - Pg_lb_hydro[g]:
                for t in range(1, nt):
                    cons_ramp_up_names = [f"cons_ramp_up_hydro_{t}_{g}"]
                    cons_ramp_up_indices = self.model_cplex.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[f"Pg_hydro_{t}_{g}", f"Pg_hydro_{t - 1}_{g}"],
                            val=[1., -1.])],
                        senses=['L'], rhs=[RUg_hydro[g]], names=cons_ramp_up_names
                    )
                    for (name, index) in zip(cons_ramp_up_names, cons_ramp_up_indices):
                        self.cons_names_to_indices[name] = index
                        self.cons_indices_to_names[index] = name
        for g in range(ng_hydro):
            if RDg_hydro[g] < Pg_ub_hydro[g] - Pg_lb_hydro[g]:
                for t in range(1, nt):
                    cons_ramp_down_names = [f"cons_ramp_down_hydro_{t}_{g}"]
                    cons_ramp_down_indices = self.model_cplex.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[f"Pg_hydro_{t - 1}_{g}", f"Pg_hydro_{t}_{g}"],
                            val=[1., -1.])],
                        senses=['L'], rhs=[RDg_hydro[g]], names=cons_ramp_down_names
                    )
                    for (name, index) in zip(cons_ramp_down_names, cons_ramp_down_indices):
                        self.cons_names_to_indices[name] = index
                        self.cons_indices_to_names[index] = name

        # constraints on Pg_gt_lb_thermal
        for g in range(ng_thermal):
            for t in range(nt):
                cons_pg_ub_names = [f"cons_pg_ub_{t}_{g}"]
                cons_pg_ub_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"Pg_gt_lb_thermal_{t}_{g}", f"ug_thermal_{t}_{g}"],
                        val=[1., - (Pg_ub_thermal[g] - Pg_lb_thermal[g])])],
                    senses=['L'], rhs=[0.], names=cons_pg_ub_names
                )
                for (name, index) in zip(cons_pg_ub_names, cons_pg_ub_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name

        # constraints on sum betag
        for t in range(nt):
            cons_sum_betag_names = [f"cons_sum_betag_{t}"]
            cons_sum_betag_indices = self.model_cplex.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[f"betag_thermal_{t}_{g}" for g in range(ng_thermal)] +
                        [f"betag_solar_{t}_{g}" for g in range(ng_solar)] +
                        [f"betag_hydro_{t}_{g}" for g in range(ng_hydro)],
                    val=np.ones(ng_thermal + ng_solar + ng_hydro))],
                senses=['E'], rhs=[1.], names=cons_sum_betag_names
            )
            for (name, index) in zip(cons_sum_betag_names, cons_sum_betag_indices):
                self.cons_names_to_indices[name] = index
                self.cons_indices_to_names[index] = name

        # constraints on betag_thermal
        for g in range(ng_thermal):
            for t in range(nt):
                cons_betag_ub_names = [f"cons_betag_ub_{t}_{g}"]
                cons_betag_ub_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_thermal_{t}_{g}", f"ug_thermal_{t}_{g}"],
                        val=[1., - 1.])],
                    senses=['L'], rhs=[0.], names=cons_betag_ub_names
                )
                for (name, index) in zip(cons_betag_ub_names, cons_betag_ub_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name

        # constraints on load balance
        for t in range(nt):
            cons_balance_names = [f"cons_balance_{t}"]
            cons_balance_indices = self.model_cplex.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[f"ug_thermal_{t}_{g}" for g in range(ng_thermal)]
                        + [f"Pg_gt_lb_thermal_{t}_{g}" for g in range(ng_thermal)]
                        + [f"Pg_solar_{t}_{g}" for g in range(ng_solar)]
                        + [f"Pg_hydro_{t}_{g}" for g in range(ng_hydro)],
                    val=np.hstack([Pg_lb_thermal, np.ones(ng_thermal + ng_solar + ng_hydro)]))],
                senses=['E'], rhs=[np.sum(Pd_survive_mean[t])], names=cons_balance_names
            )
            for (name, index) in zip(cons_balance_names, cons_balance_indices):
                self.cons_names_to_indices[name] = index
                self.cons_indices_to_names[index] = name

        # constraints on normal line flow
        for t in range(nt):
            for l in range(nl):
                cons_Fl_names = [f"cons_Fl_{t}_{l}"]
                cons_Fl_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"ug_thermal_{t}_{g}" for g in range(ng_thermal)]
                            + [f"Pg_gt_lb_thermal_{t}_{g}" for g in range(ng_thermal)]
                            + [f"Pg_solar_{t}_{g}" for g in range(ng_solar)]
                            + [f"Pg_hydro_{t}_{g}" for g in range(ng_hydro)] + [f"Fl_{t}_{l}"],
                        val=np.hstack(
                            [G_lg_thermal[l] * Pg_lb_thermal, G_lg_thermal[l], G_lg_solar[l], G_lg_hydro[l], -1.]))],
                    senses=['E'], rhs=[np.dot(G_ld[l], Pd_survive_mean[t])], names=cons_Fl_names
                )
                for (name, index) in zip(cons_Fl_names, cons_Fl_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name

        # constraints on reserve
        for t in range(nt):
            cons_reserve_names = [f"cons_reserve_{t}"]
            cons_reserve_indices = self.model_cplex.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[f"ug_thermal_{t}_{g}" for g in range(ng_thermal)]
                        + [f"Pg_solar_{t}_{g}" for g in range(ng_solar)]
                        + [f"Pg_hydro_{t}_{g}" for g in range(ng_hydro)],
                    val=np.hstack([Pg_ub_thermal, np.ones(ng_solar), np.ones(ng_hydro)]))],
                senses=['G'], rhs=[1.15 * np.sum(Pd_survive_mean[t])], names=cons_reserve_names
            )
            for (name, index) in zip(cons_reserve_names, cons_reserve_indices):
                self.cons_names_to_indices[name] = index
                self.cons_indices_to_names[index] = name

        # constraints on probabilistic Pg
        for t in range(nt):
            for g in range(ng_thermal):
                cons_pos_dPg_thermal_names = [f"cons_pos_dPg_thermal_{t}_{g}"]
                cons_pos_dPg_thermal_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_thermal_{t}_{g}", f"Pg_gt_lb_thermal_{t}_{g}", f"ug_thermal_{t}_{g}"],
                        val=[self.evar_gen_pos_coeff[t], 1., - (Pg_ub_thermal[g] - Pg_lb_thermal[g])])],
                    senses=['L'], rhs=[0.], names=cons_pos_dPg_thermal_names
                )
                for (name, index) in zip(cons_pos_dPg_thermal_names, cons_pos_dPg_thermal_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for t in range(nt):
            for g in range(ng_thermal):
                cons_neg_dPg_thermal_names = [f"cons_neg_dPg_thermal_{t}_{g}"]
                cons_neg_dPg_thermal_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_thermal_{t}_{g}", f"Pg_gt_lb_thermal_{t}_{g}"],
                        val=[self.evar_gen_neg_coeff[t], -1.])],
                    senses=['L'], rhs=[0.], names=cons_neg_dPg_thermal_names
                )
                for (name, index) in zip(cons_neg_dPg_thermal_names, cons_neg_dPg_thermal_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for t in range(nt):
            for g in range(ng_solar):
                cons_pos_dPg_solar_names = [f"cons_pos_dPg_solar_{t}_{g}"]
                cons_pos_dPg_solar_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_solar_{t}_{g}", f"Pg_solar_{t}_{g}"],
                        val=[self.evar_gen_pos_coeff[t], 1.])],
                    senses=['L'], rhs=[Pg_ub_solar[t, g]], names=cons_pos_dPg_solar_names
                )
                for (name, index) in zip(cons_pos_dPg_solar_names, cons_pos_dPg_solar_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for t in range(nt):
            for g in range(ng_solar):
                cons_neg_dPg_solar_names = [f"cons_neg_dPg_solar_{t}_{g}"]
                cons_neg_dPg_solar_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_solar_{t}_{g}", f"Pg_solar_{t}_{g}"],
                        val=[self.evar_gen_neg_coeff[t], -1.])],
                    senses=['L'], rhs=[- Pg_lb_solar[t, g]], names=cons_neg_dPg_solar_names
                )
                for (name, index) in zip(cons_neg_dPg_solar_names, cons_neg_dPg_solar_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for t in range(nt):
            for g in range(ng_hydro):
                cons_pos_dPg_hydro_names = [f"cons_pos_dPg_hydro_{t}_{g}"]
                cons_pos_dPg_hydro_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_hydro_{t}_{g}", f"Pg_hydro_{t}_{g}"],
                        val=[self.evar_gen_pos_coeff[t], 1.])],
                    senses=['L'], rhs=[Pg_ub_hydro[g]], names=cons_pos_dPg_hydro_names
                )
                for (name, index) in zip(cons_pos_dPg_hydro_names, cons_pos_dPg_hydro_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name
        for t in range(nt):
            for g in range(ng_hydro):
                cons_neg_dPg_hydro_names = [f"cons_neg_dPg_hydro_{t}_{g}"]
                cons_neg_dPg_hydro_indices = self.model_cplex.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"betag_hydro_{t}_{g}", f"Pg_hydro_{t}_{g}"],
                        val=[self.evar_gen_neg_coeff[t], -1.])],
                    senses=['L'], rhs=[- Pg_lb_hydro[g]], names=cons_neg_dPg_hydro_names
                )
                for (name, index) in zip(cons_neg_dPg_hydro_names, cons_neg_dPg_hydro_indices):
                    self.cons_names_to_indices[name] = index
                    self.cons_indices_to_names[index] = name

        self.model_cplex.objective.set_name("cost")
        # self.model_cplex.solve()
    
    def presolve(self):
        print('Start Presolving Model.')
        # presolve
        redlb, redub, rstat = self.model_cplex.advanced.basic_presolve()

        red_Pg_gt_lb_thermal_lb = np.zeros((nt, ng_thermal))
        red_Pg_gt_lb_thermal_ub = np.zeros((nt, ng_thermal))
        red_betag_thermal_lb = np.zeros((nt, ng_thermal))
        red_betag_thermal_ub = np.zeros((nt, ng_thermal))

        red_Pg_solar_lb = np.zeros((nt, ng_solar))
        red_Pg_solar_ub = np.zeros((nt, ng_solar))
        red_betag_solar_lb = np.zeros((nt, ng_solar))
        red_betag_solar_ub = np.zeros((nt, ng_solar))

        red_Pg_hydro_lb = np.zeros((nt, ng_hydro))
        red_Pg_hydro_ub = np.zeros((nt, ng_hydro))
        red_betag_hydro_lb = np.zeros((nt, ng_hydro))
        red_betag_hydro_ub = np.zeros((nt, ng_hydro))

        red_Fl_lb = np.zeros((nt, nl))
        red_Fl_ub = np.zeros((nt, nl))

        for g in range(ng_thermal):
            for t in range(nt):
                index = self.var_names_to_indices[f"Pg_gt_lb_thermal_{t}_{g}"]
                red_Pg_gt_lb_thermal_lb[t, g] = redlb[index]
                red_Pg_gt_lb_thermal_ub[t, g] = redub[index]
                index = self.var_names_to_indices[f"betag_thermal_{t}_{g}"]
                red_betag_thermal_lb[t, g] = redlb[index]
                red_betag_thermal_ub[t, g] = redub[index]
        for g in range(ng_solar):
            for t in range(nt):
                index = self.var_names_to_indices[f"Pg_solar_{t}_{g}"]
                red_Pg_solar_lb[t, g] = redlb[index]
                red_Pg_solar_ub[t, g] = redub[index]
                index = self.var_names_to_indices[f"betag_solar_{t}_{g}"]
                red_betag_solar_lb[t, g] = redlb[index]
                red_betag_solar_ub[t, g] = redub[index]
        for g in range(ng_hydro):
            for t in range(nt):
                index = self.var_names_to_indices[f"Pg_hydro_{t}_{g}"]
                red_Pg_hydro_lb[t, g] = redlb[index]
                red_Pg_hydro_ub[t, g] = redub[index]
                index = self.var_names_to_indices[f"betag_hydro_{t}_{g}"]
                red_betag_hydro_lb[t, g] = redlb[index]
                red_betag_hydro_ub[t, g] = redub[index]
        for l in range(nl):
            for t in range(nt):
                index = self.var_names_to_indices[f"Fl_{t}_{l}"]
                red_Fl_lb[t, l] = redlb[index]
                red_Fl_ub[t, l] = redub[index]

        key_Fl_lb_t_l = []
        key_Fl_ub_t_l = []
        for t in range(nt):
            model_coeff = gp.Model()
            model_coeff_betag_thermal = model_coeff.addMVar(
                shape=(ng_thermal, nl), lb=np.tile(red_betag_thermal_lb[t].reshape(-1, 1), (1, nl)),
                ub=np.tile(red_betag_thermal_ub[t].reshape(-1, 1), (1, nl)),
                vtype=GRB.CONTINUOUS, name=f"model_coeff_betag_thermal")
            model_coeff_betag_solar = model_coeff.addMVar(
                shape=(ng_solar, nl), lb=np.tile(red_betag_solar_lb[t].reshape(-1, 1), (1, nl)),
                ub=np.tile(red_betag_solar_ub[t].reshape(-1, 1), (1, nl)),
                vtype=GRB.CONTINUOUS, name=f"model_coeff_betag_solar")
            model_coeff_betag_hydro = model_coeff.addMVar(
                shape=(ng_hydro, nl), lb=np.tile(red_betag_hydro_lb[t].reshape(-1, 1), (1, nl)),
                ub=np.tile(red_betag_hydro_ub[t].reshape(-1, 1), (1, nl)),
                vtype=GRB.CONTINUOUS, name=f"model_coeff_betag_hydro")

            model_coeff.addConstr(
                model_coeff_betag_thermal.sum(axis=0) + model_coeff_betag_solar.sum(axis=0)
                + model_coeff_betag_hydro.sum(axis=0) == np.ones(nl)
            )
            model_coeff.setObjective(
                gp.quicksum([
                    G_lg_thermal[l] @ model_coeff_betag_thermal[:, l]
                    + G_lg_solar[l] @ model_coeff_betag_solar[:, l]
                    + G_lg_hydro[l] @ model_coeff_betag_hydro[:, l]
                    for l in range(nl)
                ]),
                GRB.MINIMIZE
            )
            model_coeff.setParam("OutputFlag", 0)
            model_coeff.optimize()
            min_coeff = np.array([
                G_lg_thermal[l] @ model_coeff_betag_thermal[:, l].X
                + G_lg_solar[l] @ model_coeff_betag_solar[:, l].X
                + G_lg_hydro[l] @ model_coeff_betag_hydro[:, l].X
                for l in range(nl)]).reshape(-1, 1) - G_ld
            model_coeff.setAttr("ModelSense", GRB.MAXIMIZE)
            model_coeff.optimize()
            max_coeff = np.array([
                G_lg_thermal[l] @ model_coeff_betag_thermal[:, l].X
                + G_lg_solar[l] @ model_coeff_betag_solar[:, l].X
                + G_lg_hydro[l] @ model_coeff_betag_hydro[:, l].X
                for l in range(nl)]).reshape(-1, 1) - G_ld

            esssup_pos_Fl = red_Fl_ub[t] + np.sum(
                np.maximum(max_coeff * Pd_loss_mean[t].reshape(1, -1),
                           min_coeff * (- Pd_survive_mean[t].reshape(1, -1))), axis=1)
            esssup_neg_Fl = - red_Fl_lb[t] + np.sum(
                np.maximum((- min_coeff) * Pd_loss_mean[t].reshape(1, -1),
                           (- max_coeff) * (- Pd_survive_mean[t].reshape(1, -1))), axis=1)
            for l in range(nl):
                if esssup_pos_Fl[l] > Fl_ub[l]:
                    key_Fl_ub_t_l.append((t, l))
                if esssup_neg_Fl[l] > Fl_ub[l]:
                    key_Fl_lb_t_l.append((t, l))

        print(f'len(key_Fl_lb_t_l) = {len(key_Fl_lb_t_l)}, len(key_Fl_ub_t_l) = {len(key_Fl_ub_t_l)}, '
              f'number of full Fl constraints = {nt * nl * 2}')

        return self.evar_gen_pos_coeff, self.evar_gen_neg_coeff,\
            red_betag_thermal_ub, red_betag_solar_ub, red_betag_hydro_ub, key_Fl_lb_t_l, key_Fl_ub_t_l


class MICPSolver(object):
    def __init__(
        self, evar_gen_pos_coeff, evar_gen_neg_coeff,
        red_betag_thermal_ub, red_betag_solar_ub, red_betag_hydro_ub,
        key_Fl_lb_t_l: List[Tuple[int, int]], key_Fl_ub_t_l: List[Tuple[int, int]]
    ):
        self.evar_gen_pos_coeff = evar_gen_pos_coeff
        self.evar_gen_neg_coeff = evar_gen_neg_coeff
        self.red_betag_thermal_ub = red_betag_thermal_ub
        self.red_betag_solar_ub = red_betag_solar_ub
        self.red_betag_hydro_ub = red_betag_hydro_ub
        self.key_Fl_lb_t_l = key_Fl_lb_t_l
        self.key_Fl_ub_t_l = key_Fl_ub_t_l
        self.cost_scale = 100.

        self.model = msk.Model()

        self.model._ug_thermal = self.model.variable(
            f"ug_thermal", [nt, ng_thermal], msk.Domain.binary())
        self.model._vg_thermal = self.model.variable(
            f"vg_thermal", [nt, ng_thermal], msk.Domain.binary())
        self.model._wg_thermal = self.model.variable(
            f"wg_thermal", [nt, ng_thermal], msk.Domain.binary())
        self.model._Pg_gt_lb_thermal = self.model.variable(
            f"Pg_gt_lb_thermal", [nt, ng_thermal], msk.Domain.greaterThan(0.))
        self.model._betag_thermal = self.model.variable(
            f"betag_thermal", [nt, ng_thermal], msk.Domain.greaterThan(0.))

        self.model._Pg_solar = self.model.variable(
            f"Pg_solar", [nt, ng_solar], msk.Domain.inRange(Pg_lb_solar, Pg_ub_solar))
        self.model._betag_solar = self.model.variable(
            f"betag_solar", [nt, ng_solar], msk.Domain.inRange(np.zeros((nt, ng_solar)), self.red_betag_solar_ub))

        self.model._Pg_hydro = self.model.variable(
            f"Pg_hydro", [nt, ng_hydro], msk.Domain.inRange(
                np.tile(Pg_lb_hydro.reshape(1, -1), (nt, 1)), np.tile(Pg_ub_hydro.reshape(1, -1), (nt, 1))))
        self.model._betag_hydro = self.model.variable(
            f"betag_hydro", [nt, ng_hydro], msk.Domain.inRange(np.zeros((nt, ng_hydro)), self.red_betag_hydro_ub))

        self.model._Fl = self.model.variable(
            f"Fl", [nt, nl],
            msk.Domain.inRange(- np.tile(Fl_ub.reshape(1, -1), (nt, 1)), np.tile(Fl_ub.reshape(1, -1), (nt, 1))))

        self.model._t_line_pos = self.model.variable(
            f"t_line_pos", len(self.key_Fl_ub_t_l), msk.Domain.greaterThan(0.))
        self.model._z_line_pos = self.model.variable(
            f"z_line_pos", [len(self.key_Fl_ub_t_l), nd], msk.Domain.unbounded())
        # self.model._y_line_pos = self.model.variable(
        #     f"y_line_pos", [len(self.key_Fl_ub_t_l), nd], msk.Domain.unbounded())
        self.model._v1_line_pos = self.model.variable(
            f"v1_line_pos", [len(self.key_Fl_ub_t_l), nd], msk.Domain.greaterThan(0.))
        self.model._v2_line_pos = self.model.variable(
            f"v2_line_pos", [len(self.key_Fl_ub_t_l), nd], msk.Domain.greaterThan(0.))

        self.model._t_line_neg = self.model.variable(
            f"t_line_neg", len(self.key_Fl_lb_t_l), msk.Domain.greaterThan(0.))
        self.model._z_line_neg = self.model.variable(
            f"z_line_neg", [len(self.key_Fl_lb_t_l), nd], msk.Domain.unbounded())
        # self.model._y_line_neg = self.model.variable(
        #     f"y_line_neg", [len(self.key_Fl_lb_t_l), nd], msk.Domain.unbounded())
        self.model._v1_line_neg = self.model.variable(
            f"v1_line_neg", [len(self.key_Fl_lb_t_l), nd], msk.Domain.greaterThan(0.))
        self.model._v2_line_neg = self.model.variable(
            f"v2_line_neg", [len(self.key_Fl_lb_t_l), nd], msk.Domain.greaterThan(0.))

        # constraints on startup and shutdown
        self.model._cons_logical = self.model.constraint(
            self.model._ug_thermal[1:, :] - self.model._ug_thermal[:-1, :]
            == self.model._vg_thermal[1:, :] - self.model._wg_thermal[1:, :]
        )
        self.model._cons_min_uptime_thermal = []
        for g in range(ng_thermal):
            self.model._cons_min_uptime_thermal.append([])
            for t in range(UT_gen_thermal[g] - 1, nt):
                self.model._cons_min_uptime_thermal[g].append(self.model.constraint(
                    msk.Expr.sum(self.model._vg_thermal[(t - UT_gen_thermal[g] + 1):(t + 1), g])
                    <= self.model._ug_thermal[t, g]
                ))
        self.model._cons_min_downtime_thermal = []
        for g in range(ng_thermal):
            self.model._cons_min_downtime_thermal.append([])
            for t in range(DT_gen_thermal[g] - 1, nt):
                self.model._cons_min_downtime_thermal[g].append(self.model.constraint(
                    msk.Expr.sum(self.model._wg_thermal[(t - DT_gen_thermal[g] + 1):(t + 1), g])
                    <= 1. - self.model._ug_thermal[t, g]
                ))

        # constraints on ramping
        self.model._cons_ramp_up_thermal = []
        for g in range(ng_thermal):
            if RUg_thermal[g] < Pg_ub_thermal[g] - Pg_lb_thermal[g]:
                self.model._cons_ramp_up_thermal.append(self.model.constraint(
                    self.model._Pg_gt_lb_thermal[1:, g] - self.model._Pg_gt_lb_thermal[:-1, g]
                    <= RUg_thermal[g] * self.model._ug_thermal[1:, g]
                    + (Pg_ub_thermal[g] - Pg_lb_thermal[g] - RUg_thermal[g]) * self.model._vg_thermal[1:, g]
                ))
            else:
                self.model._cons_ramp_up_thermal.append(None)
        self.model._cons_ramp_down_thermal = []
        for g in range(ng_thermal):
            if RDg_thermal[g] < Pg_ub_thermal[g] - Pg_lb_thermal[g]:
                self.model._cons_ramp_down_thermal.append(self.model.constraint(
                    self.model._Pg_gt_lb_thermal[:-1, g] - self.model._Pg_gt_lb_thermal[1:, g]
                    <= RDg_thermal[g] * self.model._ug_thermal[:-1, g]
                    + (Pg_ub_thermal[g] - Pg_lb_thermal[g] - RDg_thermal[g]) * self.model._wg_thermal[1:, g]
                ))
            else:
                self.model._cons_ramp_down_thermal.append(None)
        self.model._cons_ramp_up_hydro = []
        for g in range(ng_hydro):
            if RUg_hydro[g] < Pg_ub_hydro[g] - Pg_lb_hydro[g]:
                self.model._cons_ramp_up_hydro.append(self.model.constraint(
                    self.model._Pg_hydro[1:, g] - self.model._Pg_hydro[:-1, g] <= RUg_hydro[g]
                ))
            else:
                self.model._cons_ramp_up_hydro.append(None)
        self.model._cons_ramp_down_hydro = []
        for g in range(ng_hydro):
            if RDg_hydro[g] < Pg_ub_hydro[g] - Pg_lb_hydro[g]:
                self.model._cons_ramp_down_hydro.append(self.model.constraint(
                    self.model._Pg_hydro[:-1, g] - self.model._Pg_hydro[1:, g] <= RDg_hydro[g]
                ))
            else:
                self.model._cons_ramp_down_hydro.append(None)

        # constraints on Pg_gt_lb_thermal
        self.model._cons_Pg_ub = []
        for g in range(ng_thermal):
            self.model._cons_Pg_ub.append([])
            for t in range(nt):
                self.model._cons_Pg_ub[g].append(self.model.constraint(
                    self.model._Pg_gt_lb_thermal[t, g]
                    <= (Pg_ub_thermal[g] - Pg_lb_thermal[g]) * self.model._ug_thermal[t, g]
                ))

        # constraints on sum betag
        self.model._cons_sum_betag = self.model.constraint(
            msk.Expr.sum(self.model._betag_thermal, 1)
            + msk.Expr.sum(self.model._betag_solar, 1)
            + msk.Expr.sum(self.model._betag_hydro, 1) == np.ones(nt)
        )

        # constraints on betag_thermal
        self.model._cons_betag_ub = []
        for g in range(ng_thermal):
            self.model._cons_betag_ub.append([])
            for t in range(nt):
                self.model._cons_betag_ub[g].append(self.model.constraint(
                    self.model._betag_thermal[t, g]
                    <= self.red_betag_thermal_ub[t, g] * self.model._ug_thermal[t, g]
                ))

        # constraints on load balance
        self.model._cons_load_balance = [self.model.constraint(
            msk.Expr.dot(Pg_lb_thermal, self.model._ug_thermal[t, :])
            + msk.Expr.sum(self.model._Pg_gt_lb_thermal[t, :])
            + msk.Expr.sum(self.model._Pg_solar[t, :])
            + msk.Expr.sum(self.model._Pg_hydro[t, :])
            == np.sum(Pd_survive_mean[t])
        ) for t in range(nt)]

        # constraints on normal line flow
        self.model._cons_Fl = [self.model.constraint(
            (G_lg_thermal @ ssp.spdiags(Pg_lb_thermal, 0, ng_thermal, ng_thermal)) @ msk.Expr.reshape(
                self.model._ug_thermal[t, :], ng_thermal, 1)
            + G_lg_thermal @ msk.Expr.reshape(self.model._Pg_gt_lb_thermal[t, :], ng_thermal, 1)
            + G_lg_solar @ msk.Expr.reshape(self.model._Pg_solar[t, :], ng_solar, 1)
            + G_lg_hydro @ msk.Expr.reshape(self.model._Pg_hydro[t, :], ng_hydro, 1)
            - msk.Expr.reshape(self.model._Fl[t, :], nl, 1) == (G_ld @ Pd_survive_mean[t]).reshape(-1, 1)
        ) for t in range(nt)]

        # constraints on reserve
        self.model._cons_reserve = [self.model.constraint(
            msk.Expr.dot(Pg_ub_thermal, self.model._ug_thermal[t, :])
            + msk.Expr.sum(self.model._Pg_solar[t, :])
            + msk.Expr.sum(self.model._Pg_hydro[t, :])
            >= 1.15 * np.sum(Pd_survive_mean[t])
        ) for t in range(nt)]

        # initial relaxation constraints on dPg
        self.model._cons_pos_dPg_thermal = [self.model.constraint(
            self.evar_gen_pos_coeff[t] * self.model._betag_thermal[t, :]
            + self.model._Pg_gt_lb_thermal[t, :]
            - msk.Expr.mulElm((Pg_ub_thermal - Pg_lb_thermal).reshape(1, -1), self.model._ug_thermal[t, :])
            <= 0.
        ) for t in range(nt)]
        self.model._cons_neg_dPg_thermal = [self.model.constraint(
            self.evar_gen_neg_coeff[t] * self.model._betag_thermal[t, :]
            - self.model._Pg_gt_lb_thermal[t, :] <= 0.
        ) for t in range(nt)]

        self.model._cons_pos_dPg_solar = [self.model.constraint(
            self.evar_gen_pos_coeff[t] * self.model._betag_solar[t, :]
            + self.model._Pg_solar[t, :] - Pg_ub_solar[t, :].reshape(1, -1) <= 0.
        ) for t in range(nt)]
        self.model._cons_neg_dPg_solar = [self.model.constraint(
            self.evar_gen_neg_coeff[t] * self.model._betag_solar[t, :]
            - self.model._Pg_solar[t, :] + Pg_lb_solar[t, :].reshape(1, -1) <= 0.
        ) for t in range(nt)]

        self.model._cons_pos_dPg_hydro = [self.model.constraint(
            self.evar_gen_pos_coeff[t] * self.model._betag_hydro[t, :]
            + self.model._Pg_hydro[t, :] - Pg_ub_hydro.reshape(1, -1) <= 0.
        ) for t in range(nt)]
        self.model._cons_neg_dPg_hydro = [self.model.constraint(
            self.evar_gen_neg_coeff[t] * self.model._betag_hydro[t, :]
            - self.model._Pg_hydro[t, :] + Pg_lb_hydro.reshape(1, -1) <= 0.
        ) for t in range(nt)]

        self.model._cons_evar_line_pos_0 = []
        self.model._cons_evar_line_pos_1 = []
        self.model._cons_evar_line_pos_2 = []
        self.model._cons_evar_line_pos_3 = []
        for i, (t, l) in enumerate(self.key_Fl_ub_t_l):
            self.model._cons_evar_line_pos_0.append(self.model.constraint(
                msk.Expr.mulElm(failure_prob[t, :], msk.Expr.reshape(self.model._v1_line_pos[i, :], nd))
                + msk.Expr.mulElm(1. - failure_prob[t, :],
                                  msk.Expr.reshape(self.model._v2_line_pos[i, :], nd))
                <= msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_pos[i], nd, 0), nd)
            ))
            self.model._cons_evar_line_pos_1.append(self.model.constraint(
                msk.Expr.sum(self.model._z_line_pos[i, :])
                - math.log(alpha_obj) * self.model._t_line_pos[i]
                + msk.Expr.flatten(self.model._Fl[t, l]) - Fl_ub[l] <= 0.
            ))
            self.model._cons_evar_line_pos_2.append(self.model.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model._v1_line_pos[i, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_pos[i], nd, 0), nd, 1),
                    (- Pd_survive_mean_G_lg_thermal[t][l]) @ msk.Expr.reshape(
                        self.model._betag_thermal[t, :], ng_thermal, 1)
                    + (- Pd_survive_mean_G_lg_solar[t][l]) @ msk.Expr.reshape(
                        self.model._betag_solar[t, :], ng_solar, 1)
                    + (- Pd_survive_mean_G_lg_hydro[t][l]) @ msk.Expr.reshape(
                        self.model._betag_hydro[t, :], ng_hydro, 1)
                    - (G_ld[l] * (- Pd_survive_mean[t])).reshape(-1, 1)
                    - msk.Expr.reshape(self.model._z_line_pos[i, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            ))
            self.model._cons_evar_line_pos_3.append(self.model.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model._v2_line_pos[i, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_pos[i], nd, 0), nd, 1),
                    (Pd_loss_mean_G_lg_thermal[t][l]) @ msk.Expr.reshape(
                        self.model._betag_thermal[t, :], ng_thermal, 1)
                    + (Pd_loss_mean_G_lg_solar[t][l]) @ msk.Expr.reshape(
                        self.model._betag_solar[t, :], ng_solar, 1)
                    + (Pd_loss_mean_G_lg_hydro[t][l]) @ msk.Expr.reshape(
                        self.model._betag_hydro[t, :], ng_hydro, 1)
                    - (G_ld[l] * (Pd_loss_mean[t])).reshape(-1, 1)
                    - msk.Expr.reshape(self.model._z_line_pos[i, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            ))

        self.model._cons_evar_line_neg_0 = []
        self.model._cons_evar_line_neg_1 = []
        self.model._cons_evar_line_neg_2 = []
        self.model._cons_evar_line_neg_3 = []
        for i, (t, l) in enumerate(self.key_Fl_lb_t_l):
            self.model._cons_evar_line_neg_0.append(self.model.constraint(
                msk.Expr.mulElm(failure_prob[t, :], msk.Expr.reshape(self.model._v1_line_neg[i, :], nd))
                + msk.Expr.mulElm(1. - failure_prob[t, :],
                                  msk.Expr.reshape(self.model._v2_line_neg[i, :], nd))
                <= msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_neg[i], nd, 0), nd)
            ))
            self.model._cons_evar_line_neg_1.append(self.model.constraint(
                msk.Expr.sum(self.model._z_line_neg[i, :])
                - math.log(alpha_obj) * self.model._t_line_neg[i]
                - msk.Expr.flatten(self.model._Fl[t, l]) - Fl_ub[l] <= 0.
            ))
            self.model._cons_evar_line_neg_2.append(self.model.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model._v1_line_neg[i, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_neg[i], nd, 0), nd, 1),
                    (Pd_survive_mean_G_lg_thermal[t][l]) @ msk.Expr.reshape(
                        self.model._betag_thermal[t, :], ng_thermal, 1)
                    + (Pd_survive_mean_G_lg_solar[t][l]) @ msk.Expr.reshape(
                        self.model._betag_solar[t, :], ng_solar, 1)
                    + (Pd_survive_mean_G_lg_hydro[t][l]) @ msk.Expr.reshape(
                        self.model._betag_hydro[t, :], ng_hydro, 1)
                    - (G_ld[l] * (Pd_survive_mean[t])).reshape(-1, 1)
                    - msk.Expr.reshape(self.model._z_line_neg[i, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            ))
            self.model._cons_evar_line_neg_3.append(self.model.constraint(
                msk.Expr.hstack([
                    msk.Expr.reshape(self.model._v2_line_neg[i, :], nd, 1),
                    msk.Expr.reshape(msk.Expr.repeat(self.model._t_line_neg[i], nd, 0), nd, 1),
                    (- Pd_loss_mean_G_lg_thermal[t][l]) @ msk.Expr.reshape(
                        self.model._betag_thermal[t, :], ng_thermal, 1)
                    + (- Pd_loss_mean_G_lg_solar[t][l]) @ msk.Expr.reshape(
                        self.model._betag_solar[t, :], ng_solar, 1)
                    + (- Pd_loss_mean_G_lg_hydro[t][l]) @ msk.Expr.reshape(
                        self.model._betag_hydro[t, :], ng_hydro, 1)
                    - (G_ld[l] * (- Pd_loss_mean[t])).reshape(-1, 1)
                    - msk.Expr.reshape(self.model._z_line_neg[i, :], nd, 1)
                ]),
                msk.Domain.inPExpCone()
            ))

        self.model._objective = self.model.objective(
            "obj", msk.ObjectiveSense.Minimize,
            msk.Expr.sum(msk.Expr.vstack([
                msk.Expr.dot(gencost_thermal[:, 5] / self.cost_scale, self.model._Pg_gt_lb_thermal[t, :])
                + msk.Expr.dot((gencost_thermal[:, 5] * Pg_lb_thermal + gencost_thermal[:, 6]) / self.cost_scale,
                               self.model._ug_thermal[t, :])
                + msk.Expr.dot(gencost_thermal[:, STARTUP] / self.cost_scale, self.model._vg_thermal[t, :])
                for t in range(nt)
            ])) + msk.Expr.sum(msk.Expr.vstack([
                msk.Expr.dot(gencost_solar[:, 5] / self.cost_scale, self.model._Pg_solar[t, :])
                for t in range(nt)
            ])) + msk.Expr.sum(msk.Expr.vstack([
                msk.Expr.dot(gencost_hydro[:, 5] / self.cost_scale, self.model._Pg_hydro[t, :])
                for t in range(nt)
            ]))
        )


def write_cbf():
    presolver = Presolver()
    evar_gen_pos_coeff, evar_gen_neg_coeff, red_betag_thermal_ub, \
        red_betag_solar_ub, red_betag_hydro_ub, key_Fl_lb_t_l, key_Fl_ub_t_l = presolver.presolve()
    solver = MICPSolver(evar_gen_pos_coeff, evar_gen_neg_coeff, red_betag_thermal_ub,
                        red_betag_solar_ub, red_betag_hydro_ub, key_Fl_lb_t_l, key_Fl_ub_t_l)
    solver.model.writeTask(f"{hurricane}_{alpha_obj}.cbf")


if __name__ == "__main__":
    write_cbf()
