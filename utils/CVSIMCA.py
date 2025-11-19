import numpy as np
from itertools import product
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import BaseCrossValidator, KFold



# ------------------------------------------------------------
#  CLASSWISE K-FOLD (CV personalizzata per SIMCA)
# ------------------------------------------------------------

# class ClasswiseKFoldWithExternalVal(BaseCrossValidator):
#     """
#     Esegue KFold solo su una classe (cls_idx), ma include TUTTI gli altri campioni
#     come validation set in ogni split.
#     """
#     def __init__(self, n_splits=5, cls_idx=None, n_samples=None, shuffle=False, random_state=None):
#         self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
#         self.cls_idx = np.asarray(cls_idx)
#         self.n_samples = n_samples  # numero totale di campioni

#     def get_n_splits(self, X=None, y=None, groups=None):
#         return self.kf.get_n_splits()

#     def split(self, X, y=None, groups=None):
#         # tutti gli indici del dataset
#         all_idx = np.arange(X.shape[0])
#         # quelli NON appartenenti alla classe target
#         others = np.setdiff1d(all_idx, self.cls_idx)
#         # esegui il KFold solo sugli indici della classe
#         for train_rel, test_rel in self.kf.split(self.cls_idx):
#             train_idx = self.cls_idx[train_rel]
#             test_idx = np.concatenate([self.cls_idx[test_rel], others])
#             yield train_idx, test_idx


class ClasswiseKFoldWithExternalVal(BaseCrossValidator):
    """
    Esegue KFold solo su una classe. Puoi passare:
      - cls_idx: array di indici dei campioni della classe target, OPPURE
      - cls_label: valore dell'etichetta (p.es. 1) e verrà calcolato da y in split().
    In ogni split: train = solo subset della classe target; test = fold della classe target + TUTTI gli altri campioni.
    """
    def __init__(self, n_splits=5, cls_idx=None, cls_label=None, shuffle=False, random_state=None):
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.cls_idx = None if cls_idx is None else np.asarray(cls_idx)
        self.cls_label = cls_label

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.kf.get_n_splits()

    def split(self, X, y=None, groups=None):
        if y is None and self.cls_idx is None and self.cls_label is not None:
            raise ValueError("Per usare cls_label serve y in split(X, y).")

        # Se non ho cls_idx, ma ho cls_label, ricavo gli indici dalla y
        cls_idx = self.cls_idx
        if cls_idx is None and self.cls_label is not None:
            cls_idx = np.flatnonzero(y == self.cls_label)

        # Se qualcuno ha passato per errore uno scalare (etichetta), gestiscilo come label
        if cls_idx is not None and np.ndim(cls_idx) == 0:
            if y is None:
                raise ValueError("Hai passato uno scalare a cls_idx; serve y per ricavarne gli indici.")
            cls_idx = np.flatnonzero(y == int(cls_idx))

        if cls_idx is None or cls_idx.size == 0:
            raise ValueError("cls_idx è vuoto: nessun campione della classe target trovato.")
        if cls_idx.size < self.kf.n_splits:
            raise ValueError(f"Troppi split ({self.kf.n_splits}) rispetto ai campioni della classe target ({cls_idx.size}).")

        all_idx = np.arange(X.shape[0])
        others = np.setdiff1d(all_idx, cls_idx)

        for train_rel, test_rel in self.kf.split(cls_idx):
            train_idx = cls_idx[train_rel]
            test_idx = np.concatenate([cls_idx[test_rel], others])
            yield train_idx, test_idx


# 
def _get_simca(estimator):
    if hasattr(estimator, "_metrics_simca_conformity"):
        return estimator
    if isinstance(estimator, Pipeline):
        for _, step in reversed(list(estimator.named_steps.items())):
            if hasattr(step, "_metrics_simca_conformity"):
                return step
    raise AttributeError("Non trovo un oggetto SIMCA nell'estimator.")

def _find_ncomp_param_name(estimator):
    """Ritorna il path param per n_components (es. 'simca__n_components' in Pipeline, oppure 'n_components' su modello diretto)."""
    if isinstance(estimator, Pipeline):
        for name, step in estimator.named_steps.items():
            if hasattr(step, "_metrics_simca_conformity"):
                return f"{name}__n_components"
        raise AttributeError("Pipeline senza step SIMCA per determinare n_components.")
    else:
        return "n_components"

def cross_validate_simca_grid(
    estimator,
    X,
    y,
    cv,
    LV_min=2,
    LV_max=10,
    param_grid=None,
    refit_metric="eff",          # "eff" | "spec" | "sens"
    class_index=None,
    print_summary=True,
    store_predictions=False      # True se vuoi salvare le predizioni per ogni combinazione
):
    """
    Cross-val fedele + grid su iperparametri (SIMCA + preprocessing).
    - Se il grid include (….)n_components, NON si fa lo sweep LV_min..LV_max: si usano i valori del grid.
    - Altrimenti, si cicla su LV_min..LV_max come prima.
    - spec = media sui fold; sens = su predizioni globali; eff = sqrt(sens*spec).
    Ritorna:
      dict con:
        - 'results_df': tabella con colonne: params (dict), LV, spec, sens, eff
        - 'best_params': dict dei migliori parametri
        - 'best_score': valore max della metrica di refit
        - 'by_combo': (opzionale) predizioni per combo se store_predictions=True
    """
    if param_grid is None:
        param_grid = {}

    base_est = clone(estimator)
    ncomp_key = _find_ncomp_param_name(base_est)

    # Rileva se il grid include già n_components
    grid_includes_ncomp = any(k.endswith("n_components") for k in param_grid.keys())
    if grid_includes_ncomp:
        # In questo caso, i valori LV sono “dentro” al grid: non facciamo sweep separato.
        lv_values = None
    else:
        lv_values = list(range(LV_min, LV_max + 1))

    results_records = []
    by_combo_predictions = []  # opzionale, uno per combinazione (non per fold)

    for combo in ParameterGrid(param_grid):
        # parametri fissi per questa combinazione
        # NB: se non include n_components e facciamo sweep LV, lo imposteremo più sotto
        # Clona sempre l'estimatore “pulito”
        # (non impostiamo ancora i parametri, per poterli modificare per LV)
        # Tuttavia, param non-LV vanno messi prima del fitting.
        # Li metteremo ad ogni LV (set_params è leggero).
        predictions_this_combo = None

        # Se il grid ha già n_components, usiamo direttamente il set di valori generato da ParameterGrid
        # (che potrebbe essere un singolo valore per combo). In tal caso, lv_values_effettivi = [None] (placeholder)
        # e NON sovrascriveremo n_components.
        lv_iter = [None] if grid_includes_ncomp else lv_values

        for lv in lv_iter:
            est_lv = clone(base_est)
            # Impostiamo i parametri del combo
            est_lv.set_params(**combo)

            # Se stiamo facendo sweep LV, impostiamo n_components = lv
            if not grid_includes_ncomp:
                est_lv.set_params(**{ncomp_key: lv})

            # --- CV fedele come nella tua funzione ---
            n_samples = X.shape[0]
            n_folds = cv.get_n_splits(X, y)
            pred_vec = np.zeros(n_samples, dtype=float)  # predizioni aggregate per tutti i test index
            step_spec = np.zeros(n_folds, dtype=float)
            step_sens = np.zeros(n_folds, dtype=float)
            step_eff  = np.zeros(n_folds, dtype=float)

            splits = list(cv.split(X, y))
            last_simca_for_metrics = None

            for i, (train_idx, test_idx) in enumerate(splits, start=1):
                Xm, Lm = X[train_idx, :], y[train_idx]
                Xt, Lt = X[test_idx, :], y[test_idx]

                est_fold = clone(est_lv)
                est_fold.fit(Xm, Lm)
                try:
                    y_pred = est_fold.predict(Xt)
                except TypeError:
                    y_pred = est_fold.predict(Xt, Lt)
                y_pred = np.ravel(y_pred)

                pred_vec[test_idx] = y_pred

                simca = _get_simca(est_fold)
                ci = class_index if class_index is not None else getattr(simca, "model_class", 1)
                m = simca._metrics_simca_conformity(y_true=Lt, y_pred=y_pred, class_index=ci)

                step_spec[i-1] = m['specificity']
                step_sens[i-1] = m['sensitivity']
                step_eff[i-1]  = m.get('efficiency', float(np.sqrt(m['sensitivity'] * m['specificity'])))
                last_simca_for_metrics = simca

            # aggregazione: spec = media sui fold
            spec = float(np.mean(step_spec))
            # sens ricalcolata sull’intero dataset con predizioni aggregate
            ci = class_index if class_index is not None else getattr(last_simca_for_metrics, "model_class", 1)
            m_full = last_simca_for_metrics._metrics_simca_conformity(y_true=y, y_pred=pred_vec, class_index=ci)
            sens = float(m_full['sensitivity'])
            eff  = float(np.sqrt(sens * spec))

            # registra record
            rec = {
                "params": combo.copy(),
                "LV": (combo.get(ncomp_key) if grid_includes_ncomp else lv),
                "spec": spec,
                "sens": sens,
                "eff": eff,
            }
            results_records.append(rec)

            if store_predictions:
                by_combo_predictions.append({
                    "params": combo.copy(),
                    "LV": rec["LV"],
                    "prediction": pred_vec.copy(),
                })

    # --- scelta best in base a refit_metric ---
    metric_key = {"eff": "eff", "spec": "spec", "sens": "sens"}[refit_metric]
    # massimizziamo
    best_idx = int(np.argmax([r[metric_key] for r in results_records]))
    best_score = results_records[best_idx][metric_key]
    best_params = results_records[best_idx]["params"].copy()
    best_LV = results_records[best_idx]["LV"]

    # stampa compatta
    if print_summary:
        # ordina per (params, LV) per leggibilità
        # per evitare problemi di hashing, trasformiamo params in stringa
        def params_to_str(p): return ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
        rows = sorted(results_records, key=lambda r: (params_to_str(r["params"]), r["LV"]))
        curr = None
        for r in rows:
            pstr = params_to_str(r["params"])
            if pstr != curr:
                print("\nPARAMS:", pstr)
                curr = pstr
            print(f"  LV={r['LV']:>2} | SPEC={r['spec']:.4f} | SENS={r['sens']:.4f} | EFF={r['eff']:.4f}")
        print(f"\n[best @ {refit_metric}] LV={best_LV} | score={best_score:.4f} | params={best_params}")

    # --- refit finale del modello migliore ---
    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    # se la ricerca LV_min..LV_max era attiva, usa il best_LV per n_components
    if not any(k.endswith("n_components") for k in param_grid.keys()):
        ncomp_key = _find_ncomp_param_name(best_estimator)
        best_estimator.set_params(**{ncomp_key: best_LV})

    best_estimator.fit(X, y)

    out = {
        "results": results_records,
        "best_params": best_params,
        "best_LV": best_LV,
        "best_score": best_score,
        "best_estimator": best_estimator,   # <— aggiunto qui!
    }
    if store_predictions:
        out["by_combo"] = by_combo_predictions
    return out

import numpy as np
import matplotlib.pyplot as plt

def plot_cv(res, metric="eff", params=None, show_best=True, title=None):
    """
    Plot media ± std dei risultati di cross_validate_simca_grid.

    Parameters
    ----------
    res : dict
        Output della funzione cross_validate_simca_grid.
    metric : {'eff', 'spec', 'sens'}
        Metrica da plottare.
    params : dict, optional
        Parametri specifici da filtrare (default = best_params).
    show_best : bool, default=True
        Se True mostra la linea verticale corrispondente al best LV.
    title : str, optional
        Titolo del grafico.
    """
    results = res["results"]
    best_params = res.get("best_params", None)

    # Se non specificato, plottiamo i risultati relativi ai best_params
    if params is None and best_params is not None:
        params = best_params

    # Filtra i record che corrispondono ai parametri scelti
    def match_params(r, p):
        for k, v in p.items():
            if k not in r["params"] or r["params"][k] != v:
                return False
        return True

    selected = [r for r in results if match_params(r, params)]
    if not selected:
        raise ValueError("Nessun record trovato con i parametri specificati.")

    # Ordina per LV
    selected = sorted(selected, key=lambda r: r["LV"])
    LV = np.array([r["LV"] for r in selected])
    values = np.array([r[metric] for r in selected])

    # Non abbiamo std per fold, quindi stimiamo la dispersione locale se disponibile
    # (in futuro potresti aggiungere step_std nel dict per un plot più accurato)
    std_approx = np.zeros_like(values)

    plt.figure(figsize=(8, 5))
    plt.plot(LV, values, marker="o", color="C0", label=f"Mean CV {metric.upper()}")
    plt.fill_between(LV, values - std_approx, values + std_approx, color="C0", alpha=0.15)

    if show_best and "best_LV" in res:
        plt.axvline(res["best_LV"], color="r", linestyle="--",
                    label=f"Best LV = {res['best_LV']} ({metric} = {res['best_score']:.3f})")

    plt.xlabel("Number of latent variables (LVs)")
    plt.ylabel(metric.upper())
    plt.title(title or f"Cross-validation {metric.upper()} vs LV")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

