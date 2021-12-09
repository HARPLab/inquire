import pdb
from preference import Pref
from demonstration import Demo
from correction import Correction
from train import Session, GridTestSet
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, idx, labels):
    colors = ['r','b','g','c','m','y','k']
    for i in range(len(results)):
        series = results[i]
        x = [i+1 for i in range(len(series))]
        y = [p[idx] for p in series]
        #y = [1.0-p[0] for p in series]
        plt.plot(x, y, colors[i%len(colors)])
    plt.legend(labels)
    #plt.ylim(0,1.0)
    plt.xticks(np.arange(1, len(series)+1, 1.0))
    plt.savefig("comparison-" + str(idx) + ".png")
    plt.clf()
    #plt.show()

if __name__ == '__main__':
    series, labels = [], []
    query_count = 10 
    traj_count = 1000
    steps = 20
    w_samples = 100
    fixed_state = False
    display_traj = False
    seeds = [1,12,15,42,101] #,122,144,220,321,720] #exact number doesn't matter. used to keep random states consistent across methods.

    ts = GridTestSet(100, steps)
    comp = False
    demo = False
    pref = True
    corr = False

    if comp:
        all_runs = []
        for seed in seeds:
            print("\nBOTH: Value")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Pref,Demo]).main(ts, query_count=query_count, sample_type="value", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Both, value")

    det = True
    prob = True
    uniform = True

    if pref:
        if uniform:
            all_runs = []
            for seed in seeds:
                print("\nPREF-ONLY: Uniform")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Pref]).main(ts, query_count=query_count, sample_type="uniform", fixed_state=fixed_state, display_traj=display_traj))
            
            series.append(np.mean(all_runs,axis=0))
            labels.append("Pref-only, uniform")

        if prob:
            all_runs = []
            for seed in seeds:
                print("\nPREF-ONLY: Value-Prob")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Pref]).main(ts, query_count=query_count, sample_type="value-prob", fixed_state=fixed_state, display_traj=display_traj))
            
            series.append(np.mean(all_runs,axis=0))
            labels.append("Pref-only, value-prob")

        if det:
            all_runs = []
            for seed in seeds:
                print("\nPREF-ONLY: Value-Det")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Pref]).main(ts, query_count=query_count, sample_type="value-det", fixed_state=fixed_state, display_traj=display_traj))
            series.append(np.mean(all_runs,axis=0))
            labels.append("Pref-only, value-det")

    if demo:
        if uniform:
            all_runs = []
            for seed in seeds:
                print("\nDEMO-ONLY: Uniform")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="uniform", fixed_state=fixed_state, display_traj=display_traj))
            
            series.append(np.mean(all_runs,axis=0))
            labels.append("Demo-only, Uniform")

        if prob:
            all_runs = []
            for seed in seeds:
                print("\nDEMO-ONLY: Value-Prob")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="value-prob", fixed_state=fixed_state, display_traj=display_traj))
            
            series.append(np.mean(all_runs,axis=0))
            labels.append("Demo-only, value-prob")

        if det:
            all_runs = []
            for seed in seeds:
                print("\nDEMO-ONLY: Value-Det")
                all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="value-det", fixed_state=fixed_state, display_traj=display_traj))
            series.append(np.mean(all_runs,axis=0))
            labels.append("Demo-only, value-det")


        '''all_runs = []
        for seed in seeds:
            print("\nDEMO-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection0.5", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Demo-only, rejection-0.5")

        all_runs = []
        for seed in seeds:
            print("\nDEMO-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection0.6", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Demo-only, rejection-0.6")

        all_runs = []
        for seed in seeds:
            print("\nDEMO-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection0.7", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Demo-only, rejection-0.7")

        all_runs = []
        for seed in seeds:
            print("\nDEMO-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection0.8", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Demo-only, rejection-0.8")

        all_runs = []
        for seed in seeds:
            print("\nDEMO-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection1", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Demo-only, rejection-original")'''

        #all_runs = []
        #for seed in seeds:
        #    print("\nDEMO-ONLY: Rejection")
        #    all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection1", fixed_state=fixed_state, display_traj=display_traj))
        #series.append(np.mean(all_runs,axis=0))
        #labels.append("Demo-only, rejection-percentile")

        #all_runs = []
        #for seed in seeds:
        #    print("\nDEMO-ONLY: Rejection")
        #    all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="rejection-1", fixed_state=fixed_state, display_traj=display_traj))
        #series.append(np.mean(all_runs,axis=0))
        #labels.append("Demo-only, rejection-cdf")

        #all_runs = []
        #for seed in seeds:
        #    print("\nDEMO-ONLY: Uniform")
        #    all_runs.append(Session(w_samples, traj_count, steps, seed, int_types=[Demo]).main(ts, query_count=query_count, sample_type="uniform", fixed_state=fixed_state, display_traj=display_traj))
        #series.append(np.mean(all_runs,axis=0))
        #labels.append("Demo-only, uniform")


        '''all_runs = []
        for seed in seeds:
            print("\nPREF-ONLY: Value")
            all_runs.append(Session(w_samples, traj_count, steps, int_types=[Pref]).main(ts, query_count=query_count, sample_type="value", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Pref-only, value")

        all_runs = []
        for seed in seeds:
            print("\nPREF-ONLY: Rejection")
            all_runs.append(Session(w_samples, traj_count, steps, int_types=[Pref]).main(ts, query_count=query_count, sample_type="rejection", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Pref-only, rejection")

        all_runs = []
        for seed in seeds:
            print("\nPREF-ONLY: Uniform")
            all_runs.append(Session(w_samples, traj_count, steps, int_types=[Pref]).main(ts, query_count=query_count, sample_type="uniform", fixed_state=fixed_state, display_traj=display_traj))
        series.append(np.mean(all_runs,axis=0))
        labels.append("Pref-only, uniform")

    if corr:
        print("\nCORRECT-ONLY: Value")
        series.append(Session(w_samples, traj_count, steps, int_types=[Correction]).main(ts, query_count=query_count, sample_type="value", fixed_state=fixed_state, display_traj=display_traj))
        labels.append("Corrections-only, value")'''

    plot_results(series, 0, labels)
    plot_results(series, 1, labels)

