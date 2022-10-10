import numpy as np
import matplotlib.pyplot as plt
from attacks import AttackOptimizationInfo, UnpoisonedDataInfo

full_attack = AttackOptimizationInfo(**np.load('attack_info_full.npy', allow_pickle=True).item())
full_no_constraints = AttackOptimizationInfo(**np.load('attack_info_no_constraints.npy', allow_pickle=True).item())
full_resvar = AttackOptimizationInfo(**np.load('attack_info_resvar.npy', allow_pickle=True).item())
full_whiteness = AttackOptimizationInfo(**np.load('attack_info_whiteness.npy', allow_pickle=True).item())

fig, ax = plt.subplots(nrows=1, ncols=4)
ax[0].plot(full_attack.unpoisoned_data.U[0], label='Original')
ax[0].plot(full_attack.unpoisoned_data.U[0] + full_attack.DeltaU[-1][0], label='Constrained attack')
ax[0].plot(full_attack.unpoisoned_data.U[0] + full_no_constraints.DeltaU[-1][0], label='Unconstrainted attack')
ax[0].legend()
ax[0].grid()


ax[1].plot(np.linalg.norm(full_attack.unpoisoned_data.residuals_unpoisoned, 2, axis=0), label='Original')
ax[1].plot(np.linalg.norm(full_attack.residuals_poisoned[-1], 2, axis=0), label='Constrained attack')
ax[1].plot(np.linalg.norm(full_no_constraints.residuals_poisoned[-1], 2, axis=0), label='Unconstrainted attack')
ax[1].grid()
ax[1].legend()

ax[2].plot([x.pvalue for x in full_attack.whiteness_statistics_test_poisoned], label='Constrained attack')
#ax[2].plot([x.pvalue for x in full_no_constraints.whiteness_statistics_test_poisoned], label='Unconstrainted attack')
ax[2].grid()
ax[2].legend()

ax[3].plot(full_resvar.loss, label='Constrained attack')
#ax[3].plot(full_no_constraints.loss, label='Unconstrainted attack')
ax[3].grid()
ax[3].legend()


plt.show()


