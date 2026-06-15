"""
Regressão linear sob três paradigmas: clássico, bayesiano e machine learning.

O MESMO modelo linear (y = b0 + b1*x) é ajustado de três formas:

    1. Clássica (MQO)  -> estimador de mínimos quadrados / máxima verossimilhança
    2. Bayesiana       -> priori Gaussiana conjugada; média a posteriori (forma fechada)
    3. ML (Ridge / L2) -> regressão com penalização L2

Resultado teórico ilustrado:
    A regressão Ridge é o estimador de MÁXIMO A POSTERIORI (MAP) de uma regressão
    bayesiana com priori Gaussiana. Com priori Gaussiana + verossimilhança Gaussiana
    a posteriori é simétrica, então a média a posteriori coincide com o MAP — por
    isso os coeficientes de Bayes e Ridge batem exatamente. A correspondência dos
    hiperparâmetros é  lambda = sigma^2 / tau^2  (ruído / variância da priori).

Gera a figura `regressao_tres_paradigmas.png`.

Requisitos: numpy, scipy, scikit-learn, matplotlib
Autor: <seu nome>
Licença: MIT
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
SEED = 11
N = 12                       # amostra pequena: a regularização/shrinkage fica visível
BETA0_TRUE, BETA1_TRUE = 1.0, 2.0
SIGMA_TRUE = 2.2
ALPHA_PRIOR = 2.0            # precisão da priori Gaussiana (1 / tau^2)

COLORS = {
    "data": "#2b2b2b",
    "true": "#888888",
    "ols": "#C0392B",
    "bayes": "#1F618D",
    "ml": "#1E8449",
}

PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
}


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def gerar_dados() -> tuple[np.ndarray, np.ndarray]:
    """Gera dados sintéticos de uma relação linear com ruído Gaussiano.

    Usa a API legada `np.random.seed` (e não `default_rng`) para reproduzir
    exatamente os coeficientes reportados (MQO=2,294 ; Bayes=Ridge=1,882).
    """
    np.random.seed(SEED)
    x = np.sort(np.random.uniform(-3, 3, N))
    y = BETA0_TRUE + BETA1_TRUE * x + np.random.normal(0, SIGMA_TRUE, N)
    return x, y


# --------------------------------------------------------------------------- #
# Ajustes (um por paradigma)
# --------------------------------------------------------------------------- #
def ajuste_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mínimos quadrados ordinários. Retorna (coef, variância residual, EP da inclinação)."""
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_slope = np.sqrt(cov[1, 1])
    return beta, s2, se_slope


def ajuste_bayes(
    X: np.ndarray, y: np.ndarray, s2: float, alpha: float = ALPHA_PRIOR
) -> tuple[np.ndarray, float]:
    """
    Regressão linear bayesiana com priori Gaussiana N(0, alpha^-1 I).
    Posteriori conjugada: w ~ N(m_N, S_N). Retorna (média a posteriori, DP da inclinação).
    """
    beta_noise = 1.0 / s2                                   # precisão do ruído
    S_N = np.linalg.inv(alpha * np.eye(X.shape[1]) + beta_noise * (X.T @ X))
    m_N = beta_noise * S_N @ (X.T @ y)
    sd_slope = np.sqrt(S_N[1, 1])
    return m_N, sd_slope


def ajuste_ridge(X: np.ndarray, y: np.ndarray, s2: float, alpha: float = ALPHA_PRIOR) -> np.ndarray:
    """
    Regressão Ridge (L2). lambda = alpha * s2 faz o estimador coincidir com o MAP
    bayesiano (e, no caso Gaussiano, com a média a posteriori).
    """
    lam = alpha * s2
    ridge = Ridge(alpha=lam, fit_intercept=False)           # intercepto já está em X
    ridge.fit(X, y)
    return ridge.coef_


# --------------------------------------------------------------------------- #
# Figura
# --------------------------------------------------------------------------- #
def plotar(
    x, y, beta_ols, m_bayes, beta_ml, sd_bayes, se_ols, dof, saida: str
) -> None:
    plt.rcParams.update(PLOT_STYLE)
    x_grid = np.linspace(-3.5, 3.5, 200)
    Xg = np.column_stack([np.ones_like(x_grid), x_grid])

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)

    # Painel A: as três retas sobre os mesmos dados
    axA = fig.add_subplot(gs[0])
    axA.plot(x_grid, BETA0_TRUE + BETA1_TRUE * x_grid, color=COLORS["true"],
             ls=":", lw=2, label="Reta verdadeira", zorder=1)
    axA.plot(x_grid, Xg @ beta_ols, color=COLORS["ols"], lw=2.6, label="Clássica (MQO)")
    axA.plot(x_grid, Xg @ m_bayes, color=COLORS["bayes"], lw=2.6,
             label="Bayesiana (média posterior)")
    axA.plot(x_grid, Xg @ beta_ml, color=COLORS["ml"], lw=2.6, ls="--",
             label="ML (Ridge / L2)")
    axA.scatter(x, y, color=COLORS["data"], s=55, zorder=5,
                edgecolor="white", linewidth=1.2, label=f"Dados (n={N})")
    axA.set_title("Mesmo modelo LINEAR, três paradigmas de ajuste")
    axA.set_xlabel("x")
    axA.set_ylabel("y")
    axA.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Painel B: a inclinação beta1 sob cada paradigma
    axB = fig.add_subplot(gs[1])
    gb = np.linspace(BETA1_TRUE - 2.2, BETA1_TRUE + 2.2, 400)
    post = stats.norm.pdf(gb, m_bayes[1], sd_bayes)
    post_n = post / post.max() * 0.9
    axB.fill_between(gb, post_n, color=COLORS["bayes"], alpha=0.22,
                     label="Posteriori de $\\beta_1$ (bayes)")
    axB.plot(gb, post_n, color=COLORS["bayes"], lw=2)

    tcrit = stats.t.ppf(0.975, dof)
    axB.errorbar(beta_ols[1], 1.18, xerr=tcrit * se_ols, fmt="o",
                 color=COLORS["ols"], capsize=5, ms=9, lw=2,
                 label="MQO: ponto + IC 95%")
    axB.plot(beta_ml[1], 1.05, "D", color=COLORS["ml"], ms=10,
             label="Ridge: estimativa pontual")
    axB.axvline(BETA1_TRUE, color=COLORS["true"], ls=":", lw=2,
                label=f"$\\beta_1$ verdadeiro = {BETA1_TRUE:.1f}")
    axB.set_title("A inclinação $\\beta_1$ por paradigma")
    axB.set_xlabel("$\\beta_1$ (inclinação)")
    axB.set_yticks([])
    axB.set_ylim(0, 1.35)
    axB.legend(loc="upper left", fontsize=8.8, framealpha=0.9)

    fig.suptitle("Regressão linear nos três paradigmas — o modelo é o mesmo; "
                 "muda a filosofia", fontsize=15, fontweight="bold", y=1.0)
    fig.savefig(saida, bbox_inches="tight", facecolor="white")
    print(f"Figura salva em: {saida}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    x, y = gerar_dados()
    X = np.column_stack([np.ones_like(x), x])          # matriz de desenho [1, x]

    beta_ols, s2, se_ols = ajuste_ols(X, y)
    m_bayes, sd_bayes = ajuste_bayes(X, y, s2)
    beta_ml = ajuste_ridge(X, y, s2)

    print("Coeficientes (intercepto, inclinação):")
    print(f"  MQO   : {beta_ols[0]:.3f}, {beta_ols[1]:.3f}")
    print(f"  Bayes : {m_bayes[0]:.3f}, {m_bayes[1]:.3f}")
    print(f"  Ridge : {beta_ml[0]:.3f}, {beta_ml[1]:.3f}")
    print("  -> Bayes e Ridge coincidem (MAP = média no caso Gaussiano).")

    plotar(x, y, beta_ols, m_bayes, beta_ml, sd_bayes, se_ols,
           dof=N - 2, saida="regressao_tres_paradigmas.png")


if __name__ == "__main__":
    main()
