"""LaTeX (booktabs) tables for the paper from results/eval_paper.json.

  tab:main      outlier / direction error, all methods multi-pass + UOD, every dataset and r
  tab:accuracy  median / RMS error at r = 0.7
  tab:process   processing ablation at r = 0.7 (single vs multi-pass, without / with UOD)
  tab:model     U-Net model ablation (zero-shot vs fine-tuned), multi-pass + UOD

python scripts/make_tables_paper.py [--in results/eval_paper.json] [--out paper/tables.tex]
"""
import argparse, json, os

DS = [("synth", "Synthetic (ours)"), ("lml", "Synthetic (LML)"), ("test1", "Real: test1"),
      ("caseC", "Real: case C")]
R = [0.9, 0.7, 0.5]


def unet(d, ds):
    return f"unet:{d['_meta']['main_model'][ds]}"


def pct(x):
    return f"{100 * x:.1f}"


def cell(t, bold=False):
    s = f"{pct(t['outlier'])} / {pct(t['sign_err'])}"
    return f"\\textbf{{{s}}}" if bold else s


def tab_main(d):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Outlier rate (\%, $|\epsilon|>1$\,px) on pulse-coded double exposures. All methods see the "
         r"same image and use the same multi-pass schedule (64$\to$32$\to$32\,px, 50\,\% overlap), centre mask "
         r"and sub-pixel fit; left: no post-processing, right: the same UOD + median replacement for every "
         r"method. Bold: best non-idealised method. $^\dagger$Direction supplied by the reference (idealised). "
         r"$^\ddagger$True separate frames (upper bound, synthetic only). Brackets: 95\,\% bootstrap interval "
         r"over images.}",
         r"\label{tab:main}", r"\small", r"\begin{tabular}{llccccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Multi-pass} & \multicolumn{3}{c}{Multi-pass + UOD} & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Dataset & $r$ & Nonlin.\ AC & AC$^\dagger$ & Ours & Nonlin.\ AC & AC$^\dagger$ & Ours & True $A,B$$^\ddagger$ \\",
         r"\midrule"]
    def c(t, best=False):
        s = pct(t["outlier"])
        if "outlier_ci" in t:
            s += r"\,{\scriptsize[" + pct(t["outlier_ci"][0]) + "," + pct(t["outlier_ci"][1]) + "]}"
        return rf"\textbf{{{s}}}" if best else s
    for ds, name in DS:
        rows = [r_ for r_ in R if f"{ds}/r{r_}" in d]
        for i, r_ in enumerate(rows):
            t = d[f"{ds}/r{r_}"]; u = unet(d, ds)
            cells = []
            for suf in ("", "+uod"):
                a, o, w = t["asym" + suf], t["oracle" + suf], t[u + suf]
                cells += [c(a, a["outlier"] < w["outlier"]), c(o), c(w, w["outlier"] <= a["outlier"])]
            g = c(t["gt"]) if "gt" in t else "--"
            first = rf"\multirow{{{len(rows)}}}{{*}}{{\shortstack[l]{{{name}\\($n={t[u]['n']}$)}}}}" if i == 0 else ""
            L.append(f"{first} & {r_} & " + " & ".join(cells) + f" & {g} \\\\")
        if rows:
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def tab_direction(d):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Direction-error rate (\%, $\mathbf{u}\cdot\mathbf{u}_\mathrm{ref}<0$), same processing as "
         r"Table~\ref{tab:main}. The idealised auto-correlation has 0\,\% by construction.}",
         r"\label{tab:direction}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Nonlin.\ AC} & \multicolumn{2}{c}{Ours} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & $r$ & MP & MP+UOD & MP & MP+UOD \\", r"\midrule"]
    for ds, name in DS:
        rows = [r_ for r_ in R if f"{ds}/r{r_}" in d]
        for i, r_ in enumerate(rows):
            t = d[f"{ds}/r{r_}"]; u = unet(d, ds)
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}" if i == 0 else ""
            L.append(f"{first} & {r_} & " + " & ".join(pct(t[k]["sign_err"]) for k in
                     ("asym", "asym+uod", u, u + "+uod")) + " \\\\")
        if rows:
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tab_accuracy(d, r_=0.7):
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Median / RMS displacement error (px) at $r={r_}$, multi-pass, no post-processing.}}",
         r"\label{tab:accuracy}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Dataset & Nonlin.\ auto-corr. & Auto-corr.$^\dagger$ & U-Net + CC (ours) & True $A,B$$^\ddagger$ \\",
         r"\midrule"]
    f = lambda t: f"{t['median']:.3f} / {t['rms']:.3f}"
    for ds, name in DS:
        k = f"{ds}/r{r_}"
        if k not in d:
            continue
        t = d[k]
        g = f(t["gt"]) if "gt" in t else "--"
        L.append(f"{name} & {f(t['asym'])} & {f(t['oracle'])} & {f(t[unet(d, ds)])} & {g} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tab_process(d, r_=0.7):
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Effect of the processing on each method at $r={r_}$: outlier / direction-error rate (\%). "
         r"SP: single pass 32\,px; MP: multi-pass 64$\to$32$\to$32\,px.}",
         r"\label{tab:process}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Dataset & Method & SP & SP + UOD & MP & MP + UOD \\", r"\midrule"]
    for ds, name in DS:
        k = f"{ds}/r{r_}"
        if k not in d:
            continue
        t = d[k]; u = unet(d, ds)
        for j, (lab, sp, mp) in enumerate([("Nonlin.\\ auto-corr.", "asym_sp", "asym"),
                                           ("Auto-corr.$^\\dagger$", "oracle_sp", "oracle"),
                                           ("U-Net + CC", u.replace("unet", "unet_sp"), u)]):
            first = rf"\multirow{{3}}{{*}}{{{name}}}" if j == 0 else ""
            L.append(f"{first} & {lab} & {cell(t[sp])} & {cell(t[sp + '+uod'])} & {cell(t[mp])} & {cell(t[mp + '+uod'])} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def tab_model(d):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{U-Net trained on our synthetic images only (zero-shot) vs.\ fine-tuned on real test2 frames "
         r"(self-supervised). Outlier / direction-error rate (\%), multi-pass + UOD.}",
         r"\label{tab:model}", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
         r"Dataset & $r$ & Zero-shot & Fine-tuned \\", r"\midrule"]
    for ds, name in DS:
        rows = [r_ for r_ in R if f"{ds}/r{r_}" in d]
        for i, r_ in enumerate(rows):
            t = d[f"{ds}/r{r_}"]
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}" if i == 0 else ""
            L.append(f"{first} & {r_} & {cell(t['unet:zeroshot+uod'])} & {cell(t['unet:finetuned+uod'])} \\\\")
        if rows:
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/eval_paper.json")
    ap.add_argument("--out", default="paper/tables.tex")
    a = ap.parse_args()
    d = json.load(open(a.inp))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    txt = ("% generated by scripts/make_tables_paper.py from " + a.inp + "\n"
           "% requires \\usepackage{booktabs,multirow}\n\n" +
           "\n\n".join([tab_main(d), tab_direction(d), tab_accuracy(d), tab_process(d), tab_model(d)]) + "\n")
    open(a.out, "w").write(txt)
    print("saved", a.out)


if __name__ == "__main__":
    main()
