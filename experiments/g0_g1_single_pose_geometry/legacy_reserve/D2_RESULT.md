# D2 Fine Spatial Result

Status: **D2_NOT_SUFFICIENT__PROCEED_TO_D3_MATCHER**

| metric | D1 frozen | C2 matched | D2 fine | D2-C2 |
|---|---:|---:|---:|---:|
| proposal_top1_mean_dist_px | 3.183633 | 3.265649 | 4.556463 | +1.290814 |
| proposal_top1_pck2 | 0.533868 | 0.550313 | 0.692652 | +0.142338 |
| proposal_top1_pck4 | 0.812338 | 0.816959 | 0.826706 | +0.009747 |
| proposal_top1_pck6_4 | 0.916803 | 0.914266 | 0.856930 | -0.057335 |
| proposal_oracle_hit_rate | 0.297621 | 0.308073 | 0.427844 | +0.119771 |
| proposal_oracle_top4_recall | 0.628818 | 0.644630 | 0.765229 | +0.120599 |
| proposal_ranking_regret_mean_px | 2.274366 | 2.342006 | 3.613543 | +1.271538 |
| same_view_top1 | 0.926305 | 0.922135 | 0.919886 | -0.002250 |
| reciprocal_same_view_top1 | 0.898214 | 0.894982 | 0.892895 | -0.002088 |
| pointmap_crossview_top1 | 0.502095 | 0.504382 | 0.494517 | -0.009866 |

Family mean-distance nonworse: **3/29**

Primary gates: **1/4**. Safety: **6/6**.

D3/D4 not run. cal/sealed21/external10 untouched.
