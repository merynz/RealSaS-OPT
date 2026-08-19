from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from observable_phase import decorate_pb_observable

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--family',type=int,required=True);ap.add_argument('--episode',default='e00');ap.add_argument('--route',required=True);a=ap.parse_args()
 z=np.load(a.baseline,allow_pickle=False);NB,VB,XY=decorate_pb_observable(a.root,a.family,a.episode,a.route,z['P_A'],z['P_B']);assert np.array_equal(VB,z['V_B']);assert float(np.max(np.abs(NB-z['N_B'])))<=1e-7;print('DECORATOR_REPLAY_PASS')
if __name__=='__main__':main()
