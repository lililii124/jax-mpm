from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAM_ORDER=("E1_Pa","E2_Pa","E3_Pa","nu12","nu13","nu23","G12_Pa","G13_Pa","G23_Pa")
PARAM_LABELS=(r"$E_1$",r"$E_2$",r"$E_3$",r"$\nu_{12}$",r"$\nu_{13}$",r"$\nu_{23}$",r"$G_{12}$",r"$G_{13}$",r"$G_{23}$")

def main(argv=None):
    ap=argparse.ArgumentParser(description="Render transparent figures from saved JAX orthotropic MPM inversion outputs.")
    ap.add_argument("--result",type=Path,required=True);ap.add_argument("--loss-history",type=Path,required=True)
    ap.add_argument("--holdout",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(argv);result=json.loads(a.result.read_text(encoding="utf-8"));a.output_dir.mkdir(parents=True,exist_ok=True)
    with a.loss_history.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    iteration=np.asarray([int(r["iteration"]) for r in rows]);loss=np.asarray([float(r["loss_at_gradient"]) for r in rows])
    true_loss=float(result["loss"]["true_parameters_on_inverse_mesh"]);fit_loss=float(result["loss"]["best"])
    rel=np.asarray([100*float(result["relative_parameter_error"][k]) for k in PARAM_ORDER])
    colors=["#2878B5" if x<=5 else "#E3A018" if x<=15 else "#C44E52" for x in rel]
    fig,ax=plt.subplots(1,2,figsize=(13.2,5.25),layout="constrained")
    ax[0].semilogy(iteration,loss,"o-",color="#2878B5",markersize=3.8,linewidth=1.65,label="loss at Adam iterate")
    ax[0].axhline(true_loss,color="#55A868",linestyle="--",linewidth=1.6,label=f"truth parameters on fit mesh: {true_loss:.1f}")
    ax[0].axhline(fit_loss,color="#C44E52",linestyle=":",linewidth=1.6,label=f"best fitted loss: {fit_loss:.1f}")
    ax[0].set(xlabel="Adam iteration",ylabel="noise-whitened training objective",title="Training convergence")
    ax[0].grid(True,which="both",alpha=.22);ax[0].legend(frameon=False,fontsize=8.3)
    bars=ax[1].bar(np.arange(9),rel,color=colors,width=.72)
    ax[1].axhline(5,color="#777777",linewidth=1,linestyle="--");ax[1].axhline(15,color="#999999",linewidth=1,linestyle=":")
    ax[1].set_xticks(np.arange(9),PARAM_LABELS);ax[1].set_ylabel("absolute relative parameter error (%)")
    ax[1].set_title("Engineering constants at best fit")
    ax[1].grid(True,axis="y",alpha=.22);ax[1].set_ylim(0,max(35,float(rel.max()*1.2)))
    for b,v in zip(bars,rel):ax[1].text(b.get_x()+b.get_width()/2,b.get_height()+.5,f"{v:.1f}%",ha="center",va="bottom",fontsize=8)
    fig.suptitle("Synthetic orthotropic MAT002-SVK identification · grid 48, 2 particles/cell/axis",fontsize=12)
    fig.savefig(a.output_dir/"orthotropic_inverse_convergence_parameters.png",dpi=180,facecolor="white")
    fig.savefig(a.output_dir/"orthotropic_inverse_convergence_parameters.svg",facecolor="white")
    plt.close(fig)
    with np.load(a.holdout,allow_pickle=False) as d:
        t=d["time_s"]*1e3;ut=d["truth_u"];up=d["inferred_u"];ft=d["truth_force"];fp=d["inferred_force"];mt=d["truth_moment"];mp=d["inferred_moment"]
        positions=d["sensor_positions"]
    urms=np.sqrt(np.mean(ut**2,axis=0));sensor,component=np.unravel_index(np.argmax(urms),urms.shape)
    fcomp=int(np.argmax(np.sqrt(np.mean(ft**2,axis=0))));mcomp=int(np.argmax(np.sqrt(np.mean(mt**2,axis=0))))
    h=result["holdout"]
    fig,ax=plt.subplots(3,1,figsize=(9.5,8.2),sharex=True,layout="constrained")
    ax[0].plot(t,ut[:,sensor,component]*1e3,color="#2878B5",linewidth=1.8,label="fine-grid synthetic truth")
    ax[0].plot(t,up[:,sensor,component]*1e3,color="#C44E52",linewidth=1.55,linestyle="--",label="inverse fit")
    ax[0].set_ylabel("displacement (mm)")
    ax[0].set_title(f"Sensor {sensor+1} component {'xyz'[component]} at ({positions[sensor,0]:.3f}, {positions[sensor,1]:.3f}, {positions[sensor,2]:.3f}) m")
    ax[0].legend(frameon=False,ncol=2,fontsize=8);ax[0].grid(alpha=.2)
    ax[1].plot(t,ft[:,fcomp],color="#2878B5",linewidth=1.8,label="fine-grid synthetic truth")
    ax[1].plot(t,fp[:,fcomp],color="#C44E52",linewidth=1.55,linestyle="--",label="inverse fit")
    ax[1].set_ylabel(f"reaction force {'xyz'[fcomp]} (N)");ax[1].grid(alpha=.2)
    ax[2].plot(t,mt[:,mcomp],color="#2878B5",linewidth=1.8,label="fine-grid synthetic truth")
    ax[2].plot(t,mp[:,mcomp],color="#C44E52",linewidth=1.55,linestyle="--",label="inverse fit")
    ax[2].set_ylabel(f"reaction moment {'xyz'[mcomp]} (N m)");ax[2].set_xlabel("time (ms)");ax[2].grid(alpha=.2)
    for a0 in ax:a0.legend(frameon=False,ncol=2,fontsize=8)
    fig.suptitle("Independent synthetic holdout · mixed axial/torsion load · not used in fit\n"
                 f"Normalized RMSE: displacement {100*h['normalized_displacement_rmse']:.2f}%, force {100*h['normalized_force_rmse']:.2f}%, moment {100*h['normalized_moment_rmse']:.2f}%",fontsize=11)
    fig.savefig(a.output_dir/"orthotropic_inverse_holdout_response.png",dpi=180,facecolor="white")
    fig.savefig(a.output_dir/"orthotropic_inverse_holdout_response.svg",facecolor="white")
    plt.close(fig)
    print(json.dumps({"output_dir":str(a.output_dir.resolve()),"selected_sensor":int(sensor),"selected_component":"xyz"[int(component)],"selected_force_component":"xyz"[int(fcomp)],"selected_moment_component":"xyz"[int(mcomp)]},indent=2))

if __name__=="__main__":main()
