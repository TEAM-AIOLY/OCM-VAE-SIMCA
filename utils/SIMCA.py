import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
import scipy
from scipy.special import erfinv
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class SIMCA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components = 2, model_class = None, type: str = 'alt', t2lim = 'Fdist', t2cl = 0.95, qlim = 'jm', qcl = 0.95, dcl = 0.95, maxPC = 20, criteria = 'compl', verbose = True):
        self.n_components = n_components
        self.model_class = model_class
        self.type = type
        self.t2lim = t2lim
        self.t2cl = t2cl
        self.qlim = qlim
        self.qcl = qcl
        self.criteria = criteria
        self.dcl = dcl
        self.maxPC = maxPC
        self.metrics = {}
        self.verbose = verbose

    def fit(self, X, classes):

        if self.model_class is None:
            self.model_class = np.unique(classes)
        elif isinstance(self.model_class,int):
            self.model_class = [self.model_class]

        if not isinstance(self.n_components, list):
            self.n_components = [self.n_components]

        if len(self.n_components) == 1:
            self.n_components = [self.n_components[0]] * len(self.model_class)
        elif len(self.n_components) != len(self.model_class):
            raise ValueError("n_components length must match number of classes")
        
        if self.type == 'dd' and self.t2lim != 'chi2pom':
            print ('t2lim set as chi2pom')
            self.t2lim = 'chi2pom'
        
        if self.type == 'dd' and self.qlim != 'chi2pom':
            print ('qlim set as chi2pom')
            self.qlim = 'chi2pom'

        self._model = {}

        for i,cls in enumerate(self.model_class):
            X_cls = X[classes == cls]
            self._model[cls] = self._fit_one_class(X_cls,self.n_components[i])
        
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True            # flag esplicito (comodo)

        return self


    def _fit_one_class(self, X_cls,n_components):

        modelPCA = PCA(n_components=None, svd_solver='full')
        T = modelPCA.fit_transform(X_cls)[:, :n_components]
        P = modelPCA.components_[:n_components, :]       
        X_cls_reconstructed = T @ P + modelPCA.mean_
        residuals = X_cls - X_cls_reconstructed
        invcovT = np.linalg.pinv(np.cov(T[:,:n_components], rowvar=False))
        T2 = np.einsum("ij,jk,ik->i", T, invcovT, T)
        Q = np.sum(residuals**2, axis=1)
        T2_limit = self._Tlim(T2,n_components)
        Q_limit = self._Qlim(Q, modelPCA,n_components)
        D_limit = self._critic_distance(T2_limit, Q_limit, T2, Q, modelPCA,n_components)
        modelPCA_nc = PCA(n_components).fit(X_cls)
        if self.type == 'dd':
            T2red = self._t2dof*T2/self._t2scfact
            Qred = self._qdof*Q/self._qscfact
        else:
            T2red = T2/T2_limit
            Qred = Q/Q_limit

        return {
            "pca_model": modelPCA_nc,
            "n_components": n_components,
            "xmean": modelPCA.mean_,
            "invcovT": invcovT,
            "eigs_all": modelPCA.explained_variance_,
            "T": T,
            "P":P,
            "T2": T2,
            "Q": Q,
            "T2red": T2red,
            "Qred": Qred,
            "T2_limit": T2_limit,
            "Q_limit": Q_limit,
            "D_limit": D_limit,
            "n_samples": X_cls.shape[0],
        }
    
    def transform(self,X):
        for i in self.model_class:         
            pca = self._model[i]['pca_model']
            T = pca.transform(X)
            T2 = np.einsum("ij,jk,ik->i", T, self._model[i]["invcovT"], T)
            X_recon = pca.inverse_transform(T)
            Q = np.sum((X - X_recon)**2, axis=1)
            
            if self.type == 'dd':
                T2red = self._t2dof*T2/self._t2scfact
                Qred = self._qdof*Q/self._qscfact
            else:
                T2red = T2/self._model[i]['T2_limit']
                Qred = Q/self._model[i]['Q_limit']
            
        
        return T2,T2red,Q,Qred


    def predict(self, X, y_true=None):
        
        predictions = np.zeros((X.shape[0],len(self.model_class)))
        metrics = {}
        for i, cls in enumerate(self.model_class):
            model_info = self._model[cls]
            modelPCA = model_info["pca_model"]
            T = modelPCA.transform(X)
            X_recon = modelPCA.inverse_transform(T)
            Q = np.sum((X - X_recon)**2, axis=1)
            T2 = np.einsum("ij,jk,ik->i", T, model_info["invcovT"], T)
            if self.type == 'sim':
                t2red = T2 / model_info["T2_limit"]
                qred = Q / model_info["Q_limit"]
                dred = np.max([t2red, qred],axis=0)
            elif self.type == 'alt':
                dred = np.sqrt((T2 / model_info["T2_limit"])**2 + (Q / model_info["Q_limit"])**2)
            elif self.type == 'ci':
                t2red = T2/ model_info["T2_limit"]
                qred = Q / model_info["Q_limit"]
                dred = (t2red + qred)
            elif self.type == 'dd':
                t2red = self._t2dof * T2 / self._t2scfact 
                qred = self._qdof * Q / self._qscfact
                dred = t2red + qred
            predictions[:, i] = (dred < model_info["D_limit"])
            if y_true is not None:
                self.metrics[cls] = self._metrics_simca_conformity( y_true, predictions[:, i], cls)
                if self.verbose:
                    print(f"Sample class {cls} = {np.sum(y_true == cls)}")
                    print(f"Confusion Matrix for class {cls}:\nTP: {self.metrics[cls]['TP']}, TN: {self.metrics[cls]['TN']}, FP: {self.metrics[cls]['FP']}, FN: {self.metrics[cls]['FN']}")
                    print(f"Class {cls} - Sensitivity: {self.metrics[cls]['sensitivity']}, Specificity: {self.metrics[cls]['specificity']:.4f}, Accuracy: {self.metrics[cls]['accuracy']:.4f}, Efficiency: {self.metrics[cls]['efficiency']:.4f}")

            
        return predictions

    def _Tlim(self,T2,n_components):
        n_samples = len(T2)

        if self.t2lim == 'perc':
            T2_limit = np.percentile(T2, self.t2cl*100)

        elif self.t2lim == 'Fdistrig':
            F_value = stats.f.ppf(self.t2cl, n_components, n_samples - n_components)
            T2_limit = (n_components/n_samples)* (n_samples**2 - 1) / (n_samples - n_components) * F_value

        elif self.t2lim == 'Fdist':
            F_value = stats.f.ppf(self.t2cl, n_components, n_samples - n_components)
            T2_limit = n_components * (n_samples - 1) / (n_samples - n_components) * F_value

        elif self.t2lim == 'chi2':
            chi2_value = stats.chi2.ppf(self.t2cl, n_components)
            T2_limit = chi2_value

        elif self.t2lim == 'chi2pom':
            h0 = float(np.mean(T2))
            var_t2 = float(np.var(T2, ddof=1)) if len(T2) > 1 else 0.0
            Nh = max(int(np.round(2 * (h0**2) / var_t2)) if var_t2 > 0 else 1, 1)
            T2_limit = h0 * stats.chi2.ppf(self.t2cl, Nh) / Nh
            self._t2dof = Nh
            self._t2scfact = h0

        return T2_limit
    
    def _Qlim(self,Q,modelPCA,n_components):

        if self.qlim == 'perc':
            Q_limit = np.percentile(Q, self.qcl*100)
        elif self.qlim == 'jm':
            theta1 = modelPCA.explained_variance_[n_components:].sum()
            theta2 = (modelPCA.explained_variance_[n_components:]**2).sum()
            theta3 = (modelPCA.explained_variance_[n_components:]**3).sum()
            if theta1 == 0:
                Q_limit = 0
            else:
                h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
                h0 = max(h0, 0.001)  # avoid division by zero or negative h0
            ca    = np.sqrt(2)*erfinv(2*self.qcl-1)
            h1 = ca * np.sqrt(2 * theta2 * h0**2) / theta1
            h2 = theta2 * h0 * (h0 - 1) / (theta1**2)
            Q_limit = theta1 * (h1 + 1 + h2)**(1/h0)

        elif self.qlim == 'chi2box':
            theta1 = modelPCA.explained_variance_[n_components:].sum()
            theta2 = (modelPCA.explained_variance_[n_components:]**2).sum()
            g = theta2/theta1
            Ng = (theta1**2)/theta2
            print('here we are')
            print(theta1,theta2, g , Ng)
            Q_limit = g * stats.chi2.ppf(self.qcl, Ng)

        elif self.qlim == 'chi2pom':
            v0 = np.mean(Q)
            Nv = max(round(2 * (v0**2) / np.var(Q, ddof=1)), 1)
            Q_limit = v0 * stats.chi2.ppf(self.qcl, Nv) / Nv
            self._qdof = Nv
            self._qscfact = v0
        return Q_limit
    
    def _critic_distance(self,T2_limit,Q_limit,T2,Q, modelPCA,n_components):
        if self.type == 'sim':
            dlim = 1
        elif self.type == 'alt':
            dlim = np.sqrt(2)
        elif self.type == 'ci':
 
            theta1 = modelPCA.explained_variance_[n_components:].sum()
            theta2 = (modelPCA.explained_variance_[n_components:]**2).sum()
            tr1 = (n_components /T2_limit) + (theta1 / Q_limit)
            tr2 = (n_components/T2_limit**2) + (theta2 / Q_limit**2)
            gd = tr2 / tr1
            hd = tr1**2 / tr2
            dlim = gd * stats.chi2.ppf(self.dcl, hd)
        elif self.type == 'dd':
            dlim = stats.chi2.ppf(self.dcl, self._t2dof + self._qdof)

        return dlim
    
    def _metrics_simca_conformity(self,y_true, y_pred,class_index):
        
        true_class = (y_true == class_index).astype(int)
        TP = np.sum((y_pred == 1) & (true_class == 1))
        TN = np.sum((y_pred == 0) & (true_class == 0))
        FP = np.sum((y_pred == 1) & (true_class == 0))
        FN = np.sum((y_pred == 0) & (true_class == 1))

        # print(f"Sample class {class_index} = {np.sum(true_class)}")
        # print(f"Confusion Matrix for class {class_index}:\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        sensitivity = TP / (TP + FN) *100
        specificity = TN / (TN + FP)  *100
        accuracy = (TP + TN) / (TP + TN + FP + FN) *100
        efficiency = np.sqrt(sensitivity * specificity)

        # print(f"Class {class_index} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}, Efficiency: {efficiency:.4f}")
        metrics = {
            'sensitivity': sensitivity, 
            'specificity': specificity,
            'accuracy': accuracy,
            'efficiency': efficiency,
            'TP': TP,
            'TN':TN,
            'FP':FP,
            'FN':FN

        }

        return metrics
    
    def score(self, X, y):

        y_pred = self.predict(X,y_true=y)
        metrics = self._metrics_simca_conformity(y, y_pred,self.model_class)

        # out= metrics['efficiency']
        # out = metrics['sensitivity']
        out= metrics['specificity']


        return out
    
    def toplotT2Q(self, X, y_test):
        
        for i, cls in enumerate(self._model):
            # Trasforma i dati per la classe corrente
            T2, T2red, Q, Qred = self.transform(X)
           
            Dlim = self._model[cls]['D_limit']
            a = np.arange(0, Dlim + 0.0001, 0.0001)
            curve = np.sqrt(np.maximum(Dlim**2 - a**2, 0))
            y_color = y_test.astype(str)
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(
                T2red, Qred,
                c=y_test, cmap='viridis', s=40, edgecolor='k', linewidth=0.5, alpha= 0.7
            )
            plt.plot(a, curve, 'b-', lw=2, label=f'Confine classe {cls}')

            plt.xlabel(r"$T^2_{red}$" )
            plt.ylabel(r"$Q_{red}$")
            plt.legend(*sc.legend_elements(), title="Class")
            plt.title(rf"$T^2$ vs $Q$  Classe {cls}")
            plt.grid(True, alpha=0.3)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.show()

            return plt
            

    def toplotT2Q_iterative(self, X, y_test):
        import numpy as np
        import plotly.graph_objects as go

        figs = []
        y_color = y_test.astype(str)

        for i, cls in enumerate(self._model):
            T2, T2red, Q, Qred = self.transform(X)

            Dlim = float(self._model[cls]['D_limit'])
            a = np.linspace(0, Dlim, 1200)
            curve = np.sqrt(np.maximum(Dlim**2 - a**2, 0.0))

            x_max = max(np.max(T2red), Dlim) * 1.05 if len(T2red) else Dlim * 1.05
            y_max = max(np.max(Qred), Dlim) * 1.05 if len(Qred) else Dlim * 1.05

            fig = go.Figure()

            # Scatter di tutti i punti
            # fig.add_trace(go.Scatter(
            #     x=T2red, y=Qred, mode='markers',
            #     marker=dict(size=7, line=dict(width=0.7, color='black'), opacity=0.7),
            #     text=y_color,
            #     hovertemplate="T2red=%{x:.3f}<br>Qred=%{y:.3f}<br>Classe=%{text}<extra></extra>",
            #     name="Dati",
            # ))

            # Tracce invisibili per legenda delle classi
            for c in np.unique(y_color):
                mask = (y_color == c)
                if np.any(mask):
                    fig.add_trace(go.Scatter(
                        x=T2red[mask], y=Qred[mask], mode='markers',
                        marker=dict(size=7, line=dict(width=0.7, color='black')),
                        name=f"Class {c}",
                        showlegend=True,
                        visible=True
                    ))

            # Curva di confine (sempre in primo piano perché aggiunta per ultima)
            fig.add_trace(go.Scatter(
                x=a,
                y=curve,
                mode='lines',
                name="Decision Limit",
                hovertemplate="Limite: T2red=%{x:.3f}, Qred=%{y:.3f}<extra></extra>",
                line=dict(color='blue', width=3),
                opacity=1.0
            ))

            # Layout
            fig.update_layout(
                width=600, height=600,
                xaxis_title="T<sup>2</sup><sub>red</sub>",
                yaxis_title="Q</sup><sub>red</sub>",
                title=dict(
                    text=f"<b>T<sup>2</sup> vs Q</sup>  Class {cls}</b>",
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20, family="Arial", color="black")
                ),
                    )

            fig.update_xaxes(range=[0, x_max], zeroline=True)
            fig.update_yaxes(range=[0, y_max], zeroline=True)

            figs.append(fig)

        # Ritorna una o più figure
        return figs[0] if len(figs) == 1 else figs


   
    