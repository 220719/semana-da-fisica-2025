"""Gera o notebook Colab do minicurso (sem saídas embutidas)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Minicurso_IA_Cientifica_DRX.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def _src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.split("\n")
    return [line + ("\n" if i < len(lines) - 1 else "") for i, line in enumerate(lines)]


cells = []

cells.append(md(r"""
# MC1 — Inteligência Artificial Científica aplicada à Difração de Raios X

**Ministrante:** Dr. Anuar José Mincache  
**Afiliação:** Pós-doutorado em Física — Lund University (Suécia)  
**Evento:** Semana da Física 2026 — Universidade Estadual de Maringá (UEM)  
**Formato:** aula teórico-prática · Google Colab · sem instalação local  

**Sistema experimental:** $\mathrm{Bi}_{1-x}\mathrm{Nd}_{x}\mathrm{FeO}_{3}$ (ferrita de bismuto dopada com neodímio), $x$ entre 10% e 50%.

Este notebook é o material **reformulado** do minicurso. Os cálculos científicos (leitura, pré-processamento, extração de características e Machine Learning) são feitos por **algoritmos verificáveis**. O LLM entra **depois**, como copiloto de interpretação — **não substitui a Física**.

| Módulo | Conteúdo |
|---|---|
| 1 | O que é IA científica, ML × DL × LLM, pipeline híbrido |
| 2 | Dados reais de DRX: arquivos, leitura, visualização |
| 3 | Pré-processamento: ruído, linha de base, normalização |
| 4 | Features, catálogo de picos, Scherrer, Williamson–Hall (D e strain) |
| 5 | Pelo menos 3 modelos de regressão, métricas, overfitting |
| 6 | LLM como copiloto (interpretação e relatório) |
| 7 | Pipeline híbrido completo, do CSV ao relatório |
"""))

cells.append(md(r"""
## Como usar no Google Colab

1. Badge **Open in Colab** no `README.md`, **ou** clone (veja `docs/MANUAL.md`).
2. **CPU** basta. *Ambiente de execução → Executar tudo* (`Ctrl+F9`).
3. Figuras também são gravadas em `figuras/` (na sessão do Colab: pasta à esquerda → download).
4. Slides no navegador: [apresentação completa](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html) (não abra o `.html` na página de código do GitHub).
5. **Módulo 6 (opcional):** Secrets → `GEMINI_API_KEY`. Sem chave, o relatório local já cobre a aula.

Os CSV vêm de `data/` ou do GitHub. **Não é necessário montar o Google Drive.**
"""))

cells.append(md(r"""
---
# Módulo 1 — Introdução à Inteligência Artificial Científica

## 1.1 O que é Inteligência Artificial Científica

**Inteligência Artificial Científica** é o uso de métodos computacionais (estatística, aprendizado de máquina e modelos de linguagem) **acoplados** a um problema físico bem definido: há dados experimentais, unidades, hipóteses e **limites de validade**.

Não se trata de “pedir à IA para descobrir a estrutura cristalina”. Trata-se de:

1. organizar o experimento em um **pipeline reproduzível**;
2. extrair grandezas com **significado físico** (posição de pico, FWHM, tamanho aparente de cristalito…);
3. usar ML quando existe uma **pergunta preditiva** clara;
4. usar um LLM para **redigir e explicar** o que os algoritmos já calcularam.

Em laboratório, o risco clássico é inverter essa ordem: o texto convincente aparece antes do número conferido.

## 1.2 Machine Learning, Deep Learning e LLMs

| Família | O que faz | Neste minicurso |
|---|---|---|
| **Machine Learning** | Aprende uma função $f(X)\to y$ a partir de exemplos | Prever a dopagem de Nd a partir de características do DRX |
| **Deep Learning** | Redes com muitas camadas; em geral precisa de **muitos** dados | **Fora do escopo** — temos só 5 amostras reais |
| **LLMs** | Modelos de linguagem; geram texto condicionado a um *prompt* | Interpretar métricas e redigir o relatório |

Deep Learning não é “ML mais avançado que sempre vale a pena”. Com $n=5$, uma rede profunda só memoriza. Por isso o curso usa modelos rasos e validação honesta (Leave-One-Out).

## 1.3 Pipeline híbrido

```
Difratograma (DRX)
        ↓
Leitura dos dados
        ↓
Pré-processamento
        ↓
Extração de características
        ↓
Machine Learning
        ↓
Avaliação do modelo
        ↓
LLM  →  interpretação científica  →  relatório final
```

## 1.4 Como integrar algoritmos e LLMs sem substituir a Física

**Regra de ouro:** o LLM **não** calcula $2\theta$, FWHM, $R^2$ nem tamanho de cristalito. Ele recebe um **dossiê numérico** produzido pelo pipeline e devolve texto.

- Se o dossiê estiver errado, o relatório fica errado.
- O cientista continua **responsável** pelos números.
- Uma frase eloquente não é evidência experimental.

Essa divisão de papéis é o coração do minicurso.
"""))

cells.append(md(r"""
---
# Configuração do ambiente

No Colab, `numpy`, `pandas`, `matplotlib`, `scipy` e `scikit-learn` já vêm instalados. A célula abaixo só importa o que usaremos e define um estilo de gráfico legível em projetor (paleta distinguível para daltonismo).
"""))

cells.append(code(r"""
# ---------------------------------------------------------------------------
# Bibliotecas: cada uma tem um papel fixo no pipeline.
#   numpy / pandas  → dados numéricos e tabelas
#   matplotlib      → figuras (o aluno deve ler eixos, unidades e legendas)
#   scipy           → Savitzky–Golay, detecção de picos, linha de base esparsa
#   scikit-learn    → modelos, scaler, Leave-One-Out, métricas
# O LLM (Módulo 6) NÃO entra aqui: ele só lê um JSON pronto.
# ---------------------------------------------------------------------------
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress, pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# Paleta Okabe–Ito (melhor que o ciclo padrão do matplotlib em sala de aula)
PALETTE = {
    10: "#0072B2",
    20: "#009E73",
    30: "#E69F00",
    40: "#D55E00",
    50: "#CC79A7",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=list(PALETTE.values())),
})

FIG_DIR = Path("figuras")
FIG_DIR.mkdir(exist_ok=True)


def show_and_save(fig, name: str):
    '''Grava PNG em figuras/ e mostra na célula (Colab e local).'''
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


print("Ambiente pronto. Figuras →", FIG_DIR.resolve())
"""))

cells.append(md(r"""
---
# Módulo 2 — Conhecendo os dados experimentais

## 2.1 O que é um difratograma de pó

Na **difração de raios X em pó**, um feixe monocromático (aqui assumimos Cu Kα, $\lambda = 1{,}5406\,\text{Å}$) incide sobre um policristal. A condição de Bragg,

$$
n\lambda = 2d_{hkl}\sin\theta,
$$

seleciona os planos $\{hkl\}$ que produzem picos em $2\theta$. O arquivo experimental é, em essência, um par $(2\theta,\,I)$.

O material é $\mathrm{Bi}_{1-x}\mathrm{Nd}_{x}\mathrm{FeO}_{3}$. O Nd substitui parcialmente o Bi: isso pode alterar parâmetro de rede, distorções e, em alguns intervalos de $x$, a simetria. No DRX esperamos, *em princípio*:

- **deslocamento** de picos (mudança de $d_{hkl}$);
- **mudança de intensidade** (fator de estrutura, ocupação, textura);
- **alargamento** (cristalito, microdeformação, aberração instrumental).

Ainda não extraímos essas grandezas — primeiro precisamos **ler os arquivos corretamente**.

## 2.2 Estrutura dos arquivos

Cada CSV tem **duas colunas sem cabeçalho**: ângulo $2\theta$ (graus) e intensidade (unidades arbitrárias). Varredura típica de $\sim 10^\circ$ a $\sim 80^\circ$.

| Arquivo | Composição nominal |
|---|---|
| `Nd_10.csv` | 10% Nd |
| `Nd_20.csv` | 20% Nd |
| `Nd_30.csv` | 30% Nd |
| `Nd_40.csv` | 40% Nd |
| `Nd_50.csv` | 50% Nd |

**Correção em relação à edição anterior:** os CSVs **não** têm cabeçalho. `pd.read_csv(arquivo)` sem `header=None` transforma o **primeiro ponto experimental** em nome de coluna e **descarta** esse ponto. Também não usamos mais Google Drive: o Colab baixa os dados sozinho.
"""))

cells.append(code(r"""
# Origem dos dados:
#   1) pasta local data/ (clone do GitHub, ou notebook aberto na raiz do repo)
#   2) URL raw do GitHub (abrir só o .ipynb no Colab, sem clonar)
# Nunca assumimos cabeçalho: names= força two_theta e intensity.
REPO_RAW = "https://raw.githubusercontent.com/220719/semana-da-fisica-2025/main/data"
LOCAL_DATA = Path("data")
DOPINGS = [10, 20, 30, 40, 50]


def _read_xrd_csv(buffer_or_path) -> pd.DataFrame:
    '''Lê um difratograma (2θ, I) sem cabeçalho e devolve um DataFrame limpo.'''
    df = pd.read_csv(buffer_or_path, header=None, names=["two_theta", "intensity"])
    # valores não numéricos (linhas de comentário, se existirem) viram NaN e saem
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("two_theta").reset_index(drop=True)
    # alguns equipamentos repetem o mesmo 2θ: média evita picos “duplicados”
    if df["two_theta"].duplicated().any():
        df = df.groupby("two_theta", as_index=False)["intensity"].mean()
    return df


def load_sample(doping: int) -> pd.DataFrame:
    '''Carrega Nd_{doping}.csv do disco ou, se faltar, do GitHub.'''
    local = LOCAL_DATA / f"Nd_{doping}.csv"
    if local.exists():
        return _read_xrd_csv(local)
    url = f"{REPO_RAW}/Nd_{doping}.csv"
    return _read_xrd_csv(url)


patterns = {d: load_sample(d) for d in DOPINGS}

print("Inventário dos cinco difratogramas:\n")
for d, df in patterns.items():
    passo = np.median(np.diff(df.two_theta.to_numpy()))
    print(
        f"  Nd {d:2d}%  |  n = {len(df):5d} pontos  |  "
        f"2θ ∈ [{df.two_theta.min():.2f}, {df.two_theta.max():.2f}]°  |  "
        f"Δ(2θ) ≈ {passo:.4f}°  |  "
        f"I ∈ [{df.intensity.min():.0f}, {df.intensity.max():.0f}] u.a."
    )

patterns[10].head()
"""))

cells.append(md(r"""
## 2.4 Estatística descritiva dos arquivos

Antes de “olhar o gráfico bonito”, medimos o que cada CSV contém: número de pontos, passo angular, média/mediana/desvio da intensidade e coeficiente de variação (CV = desvio/média). O CV alto em DRX de pó é normal: a maior parte do padrão é fundo; a energia está nos picos.
"""))

cells.append(code(r"""
rows = []
for d, df in patterns.items():
    y = df.intensity.to_numpy(dtype=float)
    rows.append({
        "Nd_%": d,
        "n": len(df),
        "passo_deg": float(np.median(np.diff(df.two_theta))),
        "I_min": float(y.min()),
        "I_mediana": float(np.median(y)),
        "I_media": float(y.mean()),
        "I_max": float(y.max()),
        "I_std": float(y.std(ddof=1)),
        "CV": float(y.std(ddof=1) / y.mean()),
        "I_p95": float(np.percentile(y, 95)),
    })
stats_raw = pd.DataFrame(rows).set_index("Nd_%")
stats_raw.round(3)
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
axes[0].bar(stats_raw.index.astype(str), stats_raw.I_max, color=[PALETTE[d] for d in DOPINGS])
axes[0].set_title("Intensidade máxima")
axes[0].set_ylabel("I máx (u.a.)")
axes[1].bar(stats_raw.index.astype(str), stats_raw.I_media, color=[PALETTE[d] for d in DOPINGS])
axes[1].set_title("Intensidade média (inclui fundo)")
axes[2].bar(stats_raw.index.astype(str), stats_raw.CV, color=[PALETTE[d] for d in DOPINGS])
axes[2].set_title("CV da intensidade")
for ax in axes:
    ax.set_xlabel("Nd (%)")
fig.suptitle("Comparação estatística entre amostras (sinal bruto)", y=1.03)
fig.tight_layout()
show_and_save(fig, "02_stats_barras.png")
"""))

cells.append(code(r"""
# Histogramas: o fundo concentra a massa; a cauda direita são os picos.
fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.2), sharey=True)
for ax, d in zip(axes, DOPINGS):
    y = patterns[d].intensity
    ax.hist(y, bins=40, color=PALETTE[d], edgecolor="white", lw=0.3)
    ax.axvline(y.median(), color="#333", ls="--", lw=1)
    ax.set_title(f"{d}% Nd")
    ax.set_xlabel("I (u.a.)")
axes[0].set_ylabel("contagens")
fig.suptitle("Distribuição das intensidades (linha tracejada = mediana)", y=1.05)
fig.tight_layout()
show_and_save(fig, "02_hist_intensidade.png")
"""))

cells.append(md(r"""
## Visualização

Um único difratograma é a “impressão digital” cristalina da amostra. Em seguida comparamos as cinco dopagens com um **waterfall** (deslocamento vertical): em aula isso é mais legível que um gráfico 3D. O 3D fica como complemento. Por fim, um **mapa de intensidade** (2θ × dopagem) ajuda a ver se picos caminham com $x$.
"""))

cells.append(code(r"""
def plot_single_pattern(df: pd.DataFrame, doping: int, ax=None):
    '''Difratograma bruto: eixo x em graus, y em unidades arbitrárias do detector.'''
    ax = ax or plt.gca()
    ax.plot(df.two_theta, df.intensity, color=PALETTE[doping], lw=1.15)
    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel("Intensidade (u.a.)")
    ax.set_title(rf"DRX bruto — $\mathrm{{Bi}}_{{1-x}}\mathrm{{Nd}}_{{x}}\mathrm{{FeO}}_3$, $x$ = {doping}%")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    return ax


fig, ax = plt.subplots(figsize=(10, 4.2))
plot_single_pattern(patterns[10], 10, ax)
fig.tight_layout()
show_and_save(fig, "02_padrao_nd10.png")
"""))

cells.append(code(r"""
def plot_waterfall(patterns: dict, offset: float | None = None):
    '''Empilha os padrões no eixo y para comparar formas sem superposição total.'''
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    max_i = max(df.intensity.max() for df in patterns.values())
    # 55% da altura do pico mais intenso: suficiente para separar, sem “estourar” a figura
    step = offset if offset is not None else 0.55 * max_i

    for i, d in enumerate(sorted(patterns)):
        df = patterns[d]
        y = df.intensity.to_numpy() + i * step
        ax.plot(df.two_theta, y, color=PALETTE[d], lw=1.2, label=f"{d}% Nd")
        ax.text(df.two_theta.max() + 0.4, y[-1], f"{d}%", color=PALETTE[d], va="center", fontsize=10)

    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel("Intensidade + deslocamento (u.a.)")
    ax.set_title("Evolução dos difratogramas com a dopagem de Nd (waterfall)")
    ax.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    return fig, ax


plot_waterfall(patterns)
show_and_save(plt.gcf(), "02_waterfall.png")
"""))

cells.append(code(r"""
# Mapa 2θ × dopagem: interpolamos cada padrão numa grade comum de 2θ
# para o pcolormesh. Isso NÃO é um refinamento: é só visualização.
theta_grid = np.linspace(10, 80, 1400)
stack = np.vstack(
    [
        np.interp(theta_grid, df.two_theta, df.intensity / df.intensity.max())
        for df in (patterns[d] for d in DOPINGS)
    ]
)

fig, ax = plt.subplots(figsize=(10.5, 3.6))
im = ax.pcolormesh(
    theta_grid,
    DOPINGS,
    stack,
    shading="auto",
    cmap="cividis",
)
ax.set_yticks(DOPINGS)
ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel("Nd (%)")
ax.set_title("Mapa de intensidade normalizada (cada linha = uma amostra)")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r"$I / I_{\mathrm{max}}$")
fig.tight_layout()
show_and_save(fig, "02_mapa_intensidade.png")
"""))

cells.append(code(r"""
fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_subplot(111, projection="3d")

# stride=4: reduz pontos só para o 3D não ficar uma “fita preta”
for d, df in patterns.items():
    sl = slice(None, None, 4)
    x3 = df.two_theta.to_numpy()[sl]
    z3 = df.intensity.to_numpy()[sl]
    ax.plot(x3, np.full_like(x3, float(d), dtype=float), z3, color=PALETTE[d], lw=0.9, label=f"{d}% Nd")

ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel("Nd (%)")
ax.set_zlabel("I (u.a.)")
ax.set_title("Visão 3D complementar — a mesma informação do waterfall")
ax.view_init(elev=22, azim=-60)
ax.legend(loc="upper left")
plt.tight_layout()
show_and_save(fig, "02_visao_3d.png")
"""))

cells.append(code(r"""
# Grade comum de 2θ: permite correlacionar amostras de mesmo comprimento.
theta_c = np.linspace(10, 80, 2000)
mat = np.vstack([
    np.interp(theta_c, patterns[d].two_theta, patterns[d].intensity / patterns[d].intensity.max())
    for d in DOPINGS
])
corr = pd.DataFrame(mat.T, columns=[f"{d}%" for d in DOPINGS]).corr()

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="vlag", vmin=0.7, vmax=1, ax=axes[0], square=True)
axes[0].set_title("Correlação de Pearson entre padrões (I/Imax)")
# RMSE par a par: “quão diferentes” são dois difratogramas na grade comum
rmse = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        rmse[i, j] = np.sqrt(np.mean((mat[i] - mat[j]) ** 2))
sns.heatmap(
    pd.DataFrame(rmse, index=[f"{d}%" for d in DOPINGS], columns=[f"{d}%" for d in DOPINGS]),
    annot=True, fmt=".3f", cmap="YlGnBu_r", ax=axes[1], square=True,
)
axes[1].set_title("RMSE entre padrões normalizados")
fig.tight_layout()
show_and_save(fig, "02_corr_rmse.png")
print("Correlação média (off-diagonal):",
      float(corr.values[np.triu_indices(5, 1)].mean()))
"""))

cells.append(code(r"""
# “Registros extras” só para visualização: mistura linear entre dopagens vizinhas.
# Não são amostras novas. Mostram a tendência contínua no mapa 2θ × x.
fig, ax = plt.subplots(figsize=(10.5, 4.4))
for d in DOPINGS:
    ax.plot(theta_c, mat[DOPINGS.index(d)], color=PALETTE[d], lw=1.4, label=f"{d}% (real)")
for i, (a, b) in enumerate(zip(DOPINGS[:-1], DOPINGS[1:])):
    for w, ls in ((0.35, ":"), (0.65, ":")):
        ymix = (1 - w) * mat[DOPINGS.index(a)] + w * mat[DOPINGS.index(b)]
        ax.plot(theta_c, ymix, color="#999999", lw=0.7, ls=ls, alpha=0.85)
ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel(r"$I / I_{\mathrm{max}}$")
ax.set_title("Padrões reais + interpolações (pontilhado) entre vizinhos")
ax.legend(ncol=5, loc="upper right")
fig.tight_layout()
show_and_save(fig, "02_interpolacao_vizinhos.png")
"""))

cells.append(md(r"""
### Leitura física rápida (ainda qualitativa)

Olhe o waterfall e o mapa:

- picos **no mesmo $2\theta$** sugerem a mesma família de planos (mesma fase dominante);
- picos que **caminham** com $x$ sugerem variação de $d_{hkl}$ (lei de Bragg);
- picos que **nascem ou somem** sugerem mudança de fase ou impureza — hipótese a checar depois, não conclusão.

Interpretações quantitativas vêm **depois** do pré-processamento. Ruído e linha de base ainda estão no sinal bruto.
"""))

cells.append(md(r"""
---
# Módulo 3 — Pré-processamento dos difratogramas

Pipeline desta etapa:

1. **Suavização (Savitzky–Golay)** — ajusta um polinômio local e reduz ruído de alta frequência com menos deslocamento de pico do que uma média móvel ingênua.
2. **Linha de base (ALS)** — *asymmetric least squares* (Eilers & Boelens): penaliza curvatura da base e trata assimetricamente os resíduos para que os **picos não sejam engolidos** pela linha de fundo.
3. **Normalização** — divide pela intensidade máxima corrigida, $I \leftarrow I / I_{\max}$, para comparar **formas**.

A normalização **apaga** informação absoluta de contagens (útil para posições e formas; ruim se você quiser comparar intensidades absolutas entre medidas não calibradas). Aqui ela é didática: o ML não deve depender da escala bruta do detector.

Não “calibramos” $2\theta$ com padrão interno neste minicurso. Deslocamentos pequenos podem misturar física da amostra e erro de zero do goniômetro — vale lembrar na discussão.
"""))

cells.append(code(r"""
def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 12) -> np.ndarray:
    '''Linha de base por mínimos quadrados assimétricos (Eilers & Boelens, 2005).

    lam  -> suavidade da base (maior = mais rigida, menos sobe nos picos)
    p    -> assimetria: p pequeno faz a base seguir o fundo, nao o topo dos picos
    niter-> poucas iteracoes ja bastam neste tipo de DRX de po
    '''
    y = np.asarray(y, dtype=float)
    n = y.size
    # operador de 2ª diferença: penaliza curvatura (base “quase reta/lenta”)
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), dtype=float)
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + lam * (d2.T @ d2)).tocsc(), w * y)
        # pontos acima da base (picos) pesam pouco; pontos abaixo puxam a base
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def _savgol_window(n: int, requested: int, poly: int) -> int | None:
    '''Janela impar, menor que n, e estritamente maior que a ordem do polinomio.'''
    w = requested if requested % 2 == 1 else requested - 1
    max_odd = n if n % 2 == 1 else n - 1
    w = min(w, max_odd)
    min_w = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    if w < min_w:
        return None
    return w


def preprocess(df: pd.DataFrame, savgol_window: int = 21, savgol_poly: int = 3) -> pd.DataFrame:
    '''Devolve colunas: smooth, baseline, corrected (clip >= 0) e normalized.'''
    out = df.copy()
    y = out.intensity.to_numpy(dtype=float)
    window = _savgol_window(len(y), savgol_window, savgol_poly)
    smooth = savgol_filter(y, window_length=window, polyorder=savgol_poly) if window else y
    base = baseline_als(smooth)
    corr = np.clip(smooth - base, 0, None)  # intensidade negativa após base = artefato
    peak = corr.max() if corr.max() > 0 else 1.0
    out["smooth"] = smooth
    out["baseline"] = base
    out["corrected"] = corr
    out["normalized"] = corr / peak
    return out


processed = {d: preprocess(df) for d, df in patterns.items()}
print("Pré-processamento concluído para", list(processed))
"""))

cells.append(code(r"""
d_show = 10
df = processed[d_show]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharex=True)

axes[0].plot(df.two_theta, df.intensity, color="#888888", lw=0.8, label="Bruto")
axes[0].plot(df.two_theta, df.smooth, color=PALETTE[d_show], lw=1.2, label="Savitzky–Golay")
axes[0].plot(df.two_theta, df.baseline, color="#D55E00", lw=1.4, ls="--", label="Linha de base (ALS)")
axes[0].set_title(f"Ruído e linha de base — {d_show}% Nd")
axes[0].set_ylabel("Intensidade (u.a.)")
axes[0].legend()

axes[1].plot(df.two_theta, df.normalized, color=PALETTE[d_show], lw=1.2)
axes[1].set_title("Após correção e normalização")
axes[1].set_ylabel(r"$I / I_{\mathrm{max}}$")

fig.tight_layout()
show_and_save(fig, "03_baseline_nd10.png")
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for d, df in processed.items():
    ax.plot(df.two_theta, df.normalized, color=PALETTE[d], lw=1.15, label=f"{d}% Nd")
ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel("Intensidade normalizada")
ax.set_title("Comparação das cinco concentrações (padrões pré-processados)")
ax.legend(ncol=5, loc="upper right")
fig.tight_layout()
show_and_save(fig, "03_overlay_normalizado.png")
"""))

cells.append(code(r"""
# Painel 5 amostras + zoom no pico mais intenso (~32°) + resíduo (bruto − suave)
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
flat = axes.ravel()
for ax, d in zip(flat[:5], DOPINGS):
    df = processed[d]
    ax.plot(df.two_theta, df.normalized, color=PALETTE[d], lw=0.9)
    ax.set_title(f"{d}% Nd")
    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel(r"$I/I_{\mathrm{max}}$")
flat[5].axis("off")
fig.suptitle("Padrões pré-processados, uma amostra por painel")
fig.tight_layout()
show_and_save(fig, "03_painel_cinco.png")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3))
for d, df in processed.items():
    m = (df.two_theta >= 30) & (df.two_theta <= 34)
    axes[0].plot(df.two_theta[m], df.normalized[m], color=PALETTE[d], lw=1.3, label=f"{d}%")
axes[0].set_title("Zoom 30–34° (região do pico principal)")
axes[0].set_xlabel(r"$2\theta$ (°)")
axes[0].set_ylabel(r"$I/I_{\mathrm{max}}$")
axes[0].legend()

d0 = 10
df0 = processed[d0]
resid = df0.intensity - df0.smooth
axes[1].plot(df0.two_theta, resid, color=PALETTE[d0], lw=0.6)
axes[1].set_title(f"Resíduo bruto − Savitzky–Golay ({d0}% Nd)")
axes[1].set_xlabel(r"$2\theta$ (°)")
axes[1].set_ylabel("I (u.a.)")
fig.tight_layout()
show_and_save(fig, "03_zoom_residuo.png")
"""))

cells.append(md(r"""
---
# Módulo 4 — Extração de características (feature engineering)

Um difratograma tem milhares de pontos. O modelo de ML deste curso **não** vê o vetor inteiro (isso seria um espectro como imagem: caminho de deep learning, $n$ insuficiente). Compactamos cada amostra num **vetor curto** com leitura física.

| Característica | Significado físico aproximado |
|---|---|
| Posição do pico principal | família de planos de maior intensidade (Bragg) |
| Intensidade máxima bruta | escala do detector / quantidade difratando / textura |
| FWHM do pico principal | tamanho de cristalito + microdeformação + aberração instrumental |
| Área integrada (após base) | “massa” do envelope corrigido |
| Centroide | centro de gravidade do padrão |
| Número de picos | complexidade / possíveis fases extras |
| Média, desvio, assimetria | forma global do envelope |
| $D$ de Scherrer | tamanho **aparente** de cristalito (hipótese forte: alargamento só por tamanho) |

## Scherrer (ordem de grandeza, não Rietveld)

$$
D = \frac{K\lambda}{\beta\cos\theta}
$$

com $K \approx 0{,}9$, $\lambda = 1{,}5406\,\text{Å}$ (Cu Kα), $\beta$ = FWHM em **radianos**, $\theta$ = metade de $2\theta$ do pico.

Scherrer **não** separa alargamento instrumental nem microdeformação.

## Williamson–Hall (tamanho + strain)

O alargamento observado de cada pico, $\beta$ (FWHM em radianos), no modelo uniforme de deformação (UDM) escreve-se

$$
\beta\cos\theta = \frac{K\lambda}{D} + 4\varepsilon\sin\theta.
$$

- intercepto $\Rightarrow D$ (cristalito aparente);
- declive $\Rightarrow \varepsilon$ (microdeformação adimensional).

**Limite honesto:** não temos padrão de Si/LaB₆ neste minicurso, então **não subtraímos $\beta_{\mathrm{inst}}$**. $D$ e $\varepsilon$ são *aparentes* (incluem o instrumento). Na edição anterior o WH usava $\beta$ a partir da derivada da intensidade — isso gerava $D\sim 0{,}1\,\mathrm{nm}$, fisicamente absurdo. Aqui $\beta$ vem do FWHM de cada pico.

Cada pico vira um **registro** na tabela longa (`peak_catalog`): é o “maior número de pontos” que estes 5 CSV permitem sem inventar amostras.
"""))

cells.append(code(r"""
K_SCHERRER = 0.9
LAMBDA_ANGSTROM = 1.5406  # Cu Kα. D em Å; dividimos por 10 → nm.


def detect_peaks(two_theta, y, height_frac=0.12, prominence_frac=0.05, min_deg=0.8):
    '''Picos no padrao normalizado; FWHM via peak_widths (meia-altura).'''
    two_theta = np.asarray(two_theta, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.median(np.diff(two_theta))
    distance = max(int(min_deg / dx), 1)
    peaks, _props = find_peaks(
        y,
        height=y.max() * height_frac,
        prominence=y.max() * prominence_frac,
        distance=distance,
    )
    if len(peaks) == 0:
        return peaks, np.array([])
    widths = peak_widths(y, peaks, rel_height=0.5)
    fwhm_deg = widths[0] * dx
    return peaks, fwhm_deg


def scherrer_nm(two_theta_deg: float, fwhm_deg: float) -> float:
    '''D em nm. two_theta_deg e o 2theta do pico, nao theta.'''
    theta_rad = np.radians(two_theta_deg / 2.0)
    beta_rad = np.radians(fwhm_deg)
    if beta_rad <= 0 or np.cos(theta_rad) == 0:
        return np.nan
    d_angstrom = (K_SCHERRER * LAMBDA_ANGSTROM) / (beta_rad * np.cos(theta_rad))
    return d_angstrom / 10.0


def d_hkl_angstrom(two_theta_deg: float) -> float:
    '''d_hkl = lambda / (2 sin theta), lei de Bragg n=1.'''
    s = np.sin(np.radians(two_theta_deg / 2.0))
    return np.nan if s == 0 else LAMBDA_ANGSTROM / (2.0 * s)


def williamson_hall(two_theta_deg, fwhm_deg, min_peaks=3):
    '''Regressao linear: y = beta cos theta  vs  x = 4 sin theta.

    Devolve None se poucos picos ou intercepto nao positivo (D indefinido).
    '''
    tt = np.asarray(two_theta_deg, dtype=float)
    fw = np.asarray(fwhm_deg, dtype=float)
    ok = (fw > 0.06) & (fw < 1.2) & (tt > 18) & (tt < 75)
    tt, fw = tt[ok], fw[ok]
    if len(tt) < min_peaks:
        return None
    theta = np.radians(tt / 2.0)
    beta = np.radians(fw)
    x = 4.0 * np.sin(theta)
    y = beta * np.cos(theta)
    fit = linregress(x, y)
    intercept, slope = fit.intercept, fit.slope
    d_nm = np.nan
    if intercept > 0:
        d_nm = (K_SCHERRER * LAMBDA_ANGSTROM / intercept) / 10.0
    return {
        "x": x, "y": y, "tt": tt, "fw": fw,
        "strain": float(slope),
        "D_nm": float(d_nm) if d_nm == d_nm else np.nan,
        "intercept": float(intercept),
        "r": float(fit.rvalue),
        "p": float(fit.pvalue),
        "n": int(len(x)),
    }


def extract_features(doping: int, df: pd.DataFrame) -> dict:
    x = df.two_theta.to_numpy()
    y_corr = df.corrected.to_numpy()
    y_n = df.normalized.to_numpy()
    peaks, fwhm = detect_peaks(x, y_n)
    i_max = int(np.argmax(y_n))
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(trap(y_corr, x))
    centroid = float(trap(x * y_corr, x) / area) if area else np.nan
    main_fwhm = np.nan
    d_sch = np.nan
    d_mean = np.nan
    if len(peaks):
        j = int(np.argmax(y_n[peaks]))
        main_fwhm = float(fwhm[j])
        d_sch = float(scherrer_nm(x[peaks[j]], main_fwhm))
        d_mean = float(np.nanmean([scherrer_nm(x[p], w) for p, w in zip(peaks, fwhm)]))
    wh = williamson_hall(x[peaks], fwhm) if len(peaks) else None
    return {
        "nd_percent": doping,
        "n_points": int(len(df)),
        "n_peaks": int(len(peaks)),
        "two_theta_max": float(x[i_max]),
        "i_max_raw": float(df.intensity.max()),
        "fwhm_main_deg": main_fwhm,
        "area": area,
        "centroid_deg": centroid,
        "mean_norm": float(y_n.mean()),
        "std_norm": float(y_n.std()),
        "skew_norm": float(((y_n - y_n.mean()) ** 3).mean() / (y_n.std() ** 3 + 1e-12)),
        "scherrer_D_nm": d_sch,
        "scherrer_D_mean_nm": d_mean,
        "wh_D_nm": np.nan if wh is None else wh["D_nm"],
        "wh_strain": np.nan if wh is None else wh["strain"],
        "wh_r": np.nan if wh is None else wh["r"],
        "wh_n": 0 if wh is None else wh["n"],
        "peak_positions": x[peaks].tolist() if len(peaks) else [],
        "peak_fwhm_deg": fwhm.tolist() if len(peaks) else [],
    }


features = pd.DataFrame([extract_features(d, processed[d]) for d in DOPINGS])
feature_view = features.drop(columns=["peak_positions", "peak_fwhm_deg"])
feature_view.round(3)
"""))

cells.append(code(r"""
# Catálogo longo: um registro por pico (o que a edição antiga fazia com df_peaks).
records = []
wh_fits = {}
for d in DOPINGS:
    df = processed[d]
    x = df.two_theta.to_numpy()
    y = df.normalized.to_numpy()
    peaks, fwhm = detect_peaks(x, y)
    wh_fits[d] = williamson_hall(x[peaks], fwhm) if len(peaks) else None
    for k, (p, w) in enumerate(zip(peaks, fwhm), start=1):
        records.append({
            "nd_percent": d,
            "peak_id": k,
            "two_theta_deg": float(x[p]),
            "I_norm": float(y[p]),
            "fwhm_deg": float(w),
            "d_hkl_A": float(d_hkl_angstrom(x[p])),
            "scherrer_D_nm": float(scherrer_nm(x[p], w)),
        })
peak_catalog = pd.DataFrame(records)
print(f"{len(peak_catalog)} registros de pico  |  {peak_catalog.nd_percent.nunique()} amostras")
peak_catalog.head(12)
"""))

cells.append(code(r"""
d_show = 10
df = processed[d_show]
x = df.two_theta.to_numpy()
y = df.normalized.to_numpy()
peaks, fwhm = detect_peaks(x, y)
meta = features.loc[features.nd_percent == d_show].iloc[0]
_, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
dx = np.median(np.diff(x))
left_deg = x[0] + left_ips * dx
right_deg = x[0] + right_ips * dx

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(x, y, color=PALETTE[d_show], lw=1.15, label="Padrão normalizado")
ax.scatter(x[peaks], y[peaks], c="#D55E00", s=40, zorder=3, label="Picos")
for i, p in enumerate(peaks):
    ax.hlines(y[p] * 0.5, left_deg[i], right_deg[i], colors="#D55E00", lw=1.4, alpha=0.85)
    ax.annotate(
        f"{x[p]:.1f}°\nFWHM {fwhm[i]:.2f}°",
        (x[p], y[p]),
        textcoords="offset points", xytext=(4, 8), fontsize=8, color="#333333",
    )
ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel(r"$I / I_{\mathrm{max}}$")
ax.set_title(
    f"Picos em {d_show}% Nd  ·  máximo em {meta.two_theta_max:.2f}°  ·  "
    f"D(Scherrer pico) ≈ {meta.scherrer_D_nm:.1f} nm"
)
ax.legend()
fig.tight_layout()
show_and_save(fig, "04_picos_nd10.png")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
pairs = [
    ("n_peaks", "Nº de picos"),
    ("fwhm_main_deg", "FWHM principal (°)"),
    ("scherrer_D_nm", "D Scherrer (pico máx., nm)"),
    ("scherrer_D_mean_nm", "D Scherrer (média dos picos, nm)"),
    ("wh_D_nm", "D Williamson–Hall (nm)"),
    ("wh_strain", r"strain $\varepsilon$ (WH)"),
]
for ax, (col, ylab) in zip(axes.ravel(), pairs):
    ax.plot(features.nd_percent, features[col], marker="o", color="#0072B2", lw=1.6)
    ax.set_xlabel("Nd (%)")
    ax.set_ylabel(ylab)
    ax.set_xticks(DOPINGS)
fig.suptitle("Características vs. dopagem (incluindo cristalito e strain)", y=1.01)
fig.tight_layout()
show_and_save(fig, "04_features_vs_nd.png")
"""))

cells.append(code(r"""
pal = [PALETTE[d] for d in DOPINGS]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
sns.boxplot(data=peak_catalog, x="nd_percent", y="fwhm_deg", order=DOPINGS, palette=pal, ax=axes[0])
axes[0].set_xlabel("Nd (%)")
axes[0].set_ylabel("FWHM (°)")
axes[0].set_title("FWHM de todos os picos detectados")
sns.violinplot(data=peak_catalog, x="nd_percent", y="I_norm", order=DOPINGS, palette=pal, ax=axes[1], cut=0)
axes[1].set_xlabel("Nd (%)")
axes[1].set_ylabel(r"$I/I_{\mathrm{max}}$ no máximo do pico")
axes[1].set_title("Distribuição das intensidades de pico")
fig.tight_layout()
show_and_save(fig, "04_boxplot_violin_picos.png")
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(10.2, 4.5))
for d in DOPINGS:
    sub = peak_catalog[peak_catalog.nd_percent == d]
    ax.scatter(sub.two_theta_deg, sub.scherrer_D_nm, s=42, color=PALETTE[d], label=f"{d}%", zorder=3)
ax.set_xlabel(r"$2\theta$ (°)")
ax.set_ylabel("D Scherrer (nm)")
ax.set_title("Scherrer pico a pico — se D cair com 2θ, o strain/instrumento pesa")
ax.legend(ncol=5)
fig.tight_layout()
show_and_save(fig, "04_scherrer_por_pico.png")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 5, figsize=(14.8, 3.4), sharey=True)
for ax, d in zip(axes, DOPINGS):
    wh = wh_fits[d]
    ax.set_title(f"{d}% Nd")
    if wh is None:
        ax.text(0.5, 0.5, "poucos picos", ha="center", va="center", transform=ax.transAxes)
        continue
    ax.scatter(wh["x"], wh["y"], color=PALETTE[d], s=36, zorder=3)
    xx = np.linspace(wh["x"].min(), wh["x"].max(), 50)
    ax.plot(xx, wh["intercept"] + wh["strain"] * xx, color="#333", lw=1)
    dlab = "—" if not (wh["D_nm"] == wh["D_nm"]) else f"{wh['D_nm']:.1f} nm"
    ax.set_xlabel(r"$4\sin\theta$")
    ax.text(0.04, 0.95, f"D≈{dlab}\nε≈{wh['strain']:.4f}\nr={wh['r']:.2f}",
            transform=ax.transAxes, va="top", fontsize=8)
axes[0].set_ylabel(r"$\beta\cos\theta$")
fig.suptitle("Williamson–Hall (UDM) por amostra — β = FWHM, sem correção instrumental", y=1.08)
fig.tight_layout()
show_and_save(fig, "04_williamson_hall.png")
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(6.4, 5.2))
sc = ax.scatter(
    features.wh_strain, features.wh_D_nm,
    c=features.nd_percent, cmap="cividis", s=90, zorder=3, edgecolors="#333",
)
for _, row in features.iterrows():
    ax.annotate(f"{int(row.nd_percent)}%", (row.wh_strain, row.wh_D_nm),
                textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel(r"microstrain $\varepsilon$ (WH)")
ax.set_ylabel("D WH (nm)")
ax.set_title("Cristalito × strain (um ponto por amostra)")
fig.colorbar(sc, ax=ax, label="Nd (%)")
fig.tight_layout()
show_and_save(fig, "04_D_vs_strain.png")
"""))

cells.append(code(r"""
# Estatística entre amostras nas features (n=5: correlação é indicativa, não teste poderoso)
num = feature_view.drop(columns=["n_points"]).select_dtypes(include=[np.number])
corr_f = num.corr()
fig, ax = plt.subplots(figsize=(9.2, 7.2))
sns.heatmap(corr_f, cmap="vlag", center=0, annot=False, ax=ax)
ax.set_title("Correlação entre características (5 amostras)")
fig.tight_layout()
show_and_save(fig, "04_corr_features.png")

print("Pearson de cada feature com a % Nd (n=5):\n")
for col in num.columns:
    if col == "nd_percent":
        continue
    r, p = pearsonr(num[col].fillna(num[col].median()), num["nd_percent"])
    print(f"  {col:22s}  r = {r:+.3f}   p = {p:.3f}")
"""))

cells.append(md(r"""
Nem toda coluna “é física da dopagem”. `n_points` é artefato do arquivo. `i_max_raw` mistura detector e amostra. Scherrer e WH herdam o FWHM e a ausência de $\beta_{\mathrm{inst}}$. Intercepto WH **negativo** ⇒ $D$ indefinido: o instrumento ou o strain dominam o alargamento. O cientista escolhe o que entra no modelo.
"""))

cells.append(md(r"""
---
# Módulo 5 — Modelos de Machine Learning (regressão)

## 5.1 A pergunta

A partir das **características do DRX**, conseguimos estimar a concentração de Nd?

- **Alvo $y$:** `nd_percent` (10, 20, 30, 40, 50)
- **Entrada $X$:** colunas numéricas do quadro de características (sem o próprio alvo)

Isso é **regressão**, não classificação: o alvo é tratado como número contínuo (mesmo que o experimento só tenha cinco valores nominais).

## 5.2 Por que não fazer split 80/20

Com **cinco amostras reais**, um teste de 20% teria **um** ponto. Qualquer $R^2$ nesse teste é loteria. Usamos **Leave-One-Out (LOO)**: treina em 4, testa na que ficou de fora, repete cinco vezes. É o protocolo honesto — e costuma doer no $R^2$.

## 5.3 Métricas

- **$R^2$:** fração da variância de $y$ explicada. No LOO pode ser **negativa** (pior que prever a média).
- **MAE:** erro absoluto médio, em **pontos percentuais** de Nd.
- **RMSE:** mesma unidade, penaliza erros grandes.

Comparamos sempre **ajuste in-sample** versus **LOO**. A diferença é a cara do overfitting.

## 5.4 Quatro modelos (pelo menos três, como na edição anterior)

- **Regressão linear** — hiperplano; $p \approx n$ é perigoso.
- **k-NN** ($k=2$) — prediz pela média dos vizinhos no espaço de features.
- **SVR (RBF)** — superfície suave; com $n=5$ também memoriza.
- **Random Forest** — flexível; o exemplo clássico de overfitting neste dataset.
"""))

cells.append(code(r"""
TARGET = "nd_percent"
# n_points não entra: é artefato do arquivo, não da física da dopagem
X_cols = [
    "n_peaks",
    "two_theta_max",
    "i_max_raw",
    "fwhm_main_deg",
    "area",
    "centroid_deg",
    "mean_norm",
    "std_norm",
    "skew_norm",
    "scherrer_D_nm",
    "scherrer_D_mean_nm",
    "wh_D_nm",
    "wh_strain",
]

X = feature_view[X_cols].to_numpy(dtype=float)
y = feature_view[TARGET].to_numpy(dtype=float)

# Se FWHM/Scherrer vier NaN (pico não encontrado), imputamos a média da coluna.
# Com n=5 isso é frágil — e deve aparecer no relatório.
col_mean = np.nanmean(X, axis=0)
col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])


def eval_model(name, estimator):
    '''Pipeline: padroniza X (media 0, variancia 1) e depois o modelo.

    Sem scaler, Ridge/linear misturam unidades (graus vs u.a. vs nm).
    y_loo = predicao de cada amostra quando ela ficou FORA do treino.
    y_in  = predicao apos treinar nas 5 (otimista, so para contrastar).
    '''
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    y_loo = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    pipe.fit(X, y)
    y_in = pipe.predict(X)
    metrics = {
        "modelo": name,
        "R2_ajuste": r2_score(y, y_in),
        "MAE_ajuste": mean_absolute_error(y, y_in),
        "RMSE_ajuste": mean_squared_error(y, y_in) ** 0.5,
        "R2_LOO": r2_score(y, y_loo),
        "MAE_LOO": mean_absolute_error(y, y_loo),
        "RMSE_LOO": mean_squared_error(y, y_loo) ** 0.5,
    }
    return metrics, y_in, y_loo, pipe


results = []
preds = {}
for name, est in [
    ("Regressão linear", LinearRegression()),
    ("k-NN (k=2)", KNeighborsRegressor(n_neighbors=2)),
    ("SVR (RBF)", SVR(kernel="rbf", C=10.0, gamma="scale")),
    ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, max_depth=3)),
]:
    m, y_in, y_loo, fitted = eval_model(name, est)
    results.append(m)
    preds[name] = {"ajuste": y_in, "LOO": y_loo, "pipe": fitted}

metrics_df = pd.DataFrame(results).set_index("modelo").round(3)
print(f"n_amostras = {len(y)}  |  n_features = {X.shape[1]}  (regime perigoso: p > n)\n")
metrics_df
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True, sharey=True)
lims = [5, 55]
for ax, mode in zip(axes, ["ajuste", "LOO"]):
    for name, blob in preds.items():
        ax.scatter(y, blob[mode], s=78, label=name, zorder=3)
    ax.plot(lims, lims, ls="--", c="0.4", lw=1, label=r"ideal $y=x$")
    ax.set_title("Treino nas 5 amostras (otimista)" if mode == "ajuste" else "Leave-One-Out (honesto)")
    ax.set_xlabel("Nd real (%)")
    ax.set_ylabel("Nd predito (%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xticks(DOPINGS)
    ax.set_yticks(DOPINGS)
axes[1].legend(loc="upper left", fontsize=9)
fig.suptitle("Overfitting: o ajuste “bonito” à esquerda não precisa sobreviver ao LOO", y=1.03)
fig.tight_layout()
show_and_save(fig, "05_ajuste_vs_loo.png")
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
metrics_df[["R2_ajuste", "R2_LOO"]].plot(kind="bar", ax=axes[0], color=["#0072B2", "#D55E00"], rot=20)
axes[0].set_title("R²")
axes[0].axhline(0, color="0.5", lw=0.8)
metrics_df[["MAE_ajuste", "MAE_LOO"]].plot(kind="bar", ax=axes[1], color=["#0072B2", "#D55E00"], rot=20)
axes[1].set_title("MAE (p.p. de Nd)")
metrics_df[["RMSE_ajuste", "RMSE_LOO"]].plot(kind="bar", ax=axes[2], color=["#0072B2", "#D55E00"], rot=20)
axes[2].set_title("RMSE (p.p. de Nd)")
for ax in axes:
    ax.set_xlabel("")
    ax.legend(fontsize=8)
fig.suptitle("Quatro modelos — ajuste vs Leave-One-Out", y=1.05)
fig.tight_layout()
show_and_save(fig, "05_metricas_barras.png")
"""))

cells.append(code(r"""
comp = pd.DataFrame({"Nd real": y})
for name, blob in preds.items():
    comp[f"{name} (LOO)"] = np.round(blob["LOO"], 2)
comp
"""))

cells.append(md(r"""
## 5.5 Bases pequenas, dados sintéticos e overfitting

Cinco pontos no espaço de ~10 características é o regime em que **qualquer modelo flexível memoriza**. O Random Forest com $R^2$ alto no ajuste e ruim no LOO é o exemplo em sala.

**Dados sintéticos** (ruído multiplicativo em $X$, ruído pequeno em $y$) aumentam $n$, mas **não adicionam física nova**: são ecos das mesmas cinco medidas. Servem para ensinar o fluxo treino/teste, **não** para reivindicar um modelo publicável.

A célula abaixo gera sintéticos **só para a demonstração pedagógica**. O aviso vai no JSON do LLM.
"""))

cells.append(code(r"""
rng = np.random.default_rng(42)
n_per = 25          # 25 ecos artificiais de cada amostra real
noise = 0.04        # 4% de ruído relativo em cada feature

X_syn, y_syn = [], []
for i, yi in enumerate(y):
    for _ in range(n_per):
        X_syn.append(X[i] * rng.normal(1.0, noise, size=X.shape[1]))
        y_syn.append(yi + rng.normal(0, 0.4))
X_syn = np.vstack(X_syn)
y_syn = np.asarray(y_syn)

X_tr, X_te, y_tr, y_te = train_test_split(X_syn, y_syn, test_size=0.25, random_state=42)
syn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)),
])
syn_pipe.fit(X_tr, y_tr)
y_te_pred = syn_pipe.predict(X_te)

syn_metrics = {
    "R2_teste_sintetico": float(r2_score(y_te, y_te_pred)),
    "MAE_teste_sintetico": float(mean_absolute_error(y_te, y_te_pred)),
    "RMSE_teste_sintetico": float(mean_squared_error(y_te, y_te_pred) ** 0.5),
    "n_treino": int(len(y_tr)),
    "n_teste": int(len(y_te)),
}
print(json.dumps(syn_metrics, indent=2))

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.scatter(y_te, y_te_pred, s=28, alpha=0.65, color="#0072B2")
lims = [min(y_te.min(), y_te_pred.min()) - 1, max(y_te.max(), y_te_pred.max()) + 1]
ax.plot(lims, lims, "--", c="0.4")
ax.set_xlabel("Nd sintético (teste)")
ax.set_ylabel("Nd predito")
ax.set_title("RF em dados sintéticos — didático, não é validação experimental")
ax.set_aspect("equal")
fig.tight_layout()
show_and_save(fig, "05_sintetico.png")
"""))

cells.append(md(r"""
---
# Módulo 6 — LLM como copiloto científico

O modelo de linguagem recebe um **JSON com resultados já calculados**. Ele deve:

- interpretar as métricas em linguagem de relatório;
- explicar $R^2$, MAE e RMSE **neste experimento** (não uma definição de livro solta);
- resumir o que os difratogramas mostraram;
- **recusar-se** a inventar picos, fases ou valores que não estejam no JSON.

Se não houver chave de API, usamos um gerador **local e determinístico** — ainda assim baseado só nos números do pipeline — para a aula não parar.

Instalação opcional no Colab (só se for usar Gemini/OpenAI):

```
%pip install -q google-genai openai
```
"""))

cells.append(code(r"""
dossie = {
    "material": "Bi1-xNdxFeO3",
    "tecnica": "Difração de raios X (pó)",
    "n_amostras_reais": int(len(DOPINGS)),
    "dopagens_percent": DOPINGS,
    "leitura": {
        nome: {
            "n_pontos": int(len(df)),
            "two_theta_min": float(df.two_theta.min()),
            "two_theta_max": float(df.two_theta.max()),
        }
        for nome, df in ((f"Nd_{d}", patterns[d]) for d in DOPINGS)
    },
    "caracteristicas": feature_view.round(4).to_dict(orient="records"),
    "n_registros_de_pico": int(len(peak_catalog)),
    "williamson_hall": feature_view[["nd_percent", "wh_D_nm", "wh_strain", "wh_r", "wh_n"]].round(4).to_dict(orient="records"),
    "ml_loo": metrics_df.reset_index().to_dict(orient="records"),
    "previsoes_LOO": {
        name: {"real": y.tolist(), "predito": np.round(blob["LOO"], 3).tolist()}
        for name, blob in preds.items()
    },
    "dados_sinteticos": {
        **syn_metrics,
        "aviso": "Sintéticos são ruído/interpolação das 5 medidas reais; não substituem experimento.",
    },
    "instrucoes_ao_llm": [
        "Não invente números que não estejam neste JSON.",
        "Deixe claro que o LLM não calculou FWHM, Scherrer, Williamson-Hall nem as métricas.",
        "Discuta overfitting e o regime n=5.",
        "Escreva em português, tom de relatório científico de minicurso (não paper Nature).",
    ],
}

print(json.dumps({k: dossie[k] for k in ["material", "n_amostras_reais", "ml_loo"]}, indent=2, ensure_ascii=False))
"""))

cells.append(code(r"""
SYSTEM = (
    "Você é um copiloto científico em um minicurso de Física/ciência de dados. "
    "Você NÃO calcula difração, FWHM, Scherrer, Williamson-Hall nem métricas de ML. "
    "Você apenas interpreta o JSON fornecido pelo pipeline verificável. "
    "Se algo não estiver no JSON, diga que não consta. "
    "Estruture a resposta em: (1) dados e pré-processamento, (2) características físicas, "
    "(3) modelos e métricas, (4) limites e overfitting, (5) síntese para os alunos. "
    "Português do Brasil, claro, sem jargão vazio."
)


def relatorio_local(dossie: dict) -> str:
    '''Fallback deterministico: so interpola os numeros ja calculados em texto.'''
    loo = {row["modelo"]: row for row in dossie["ml_loo"]}
    linhas = [
        "# Relatório científico (gerado localmente — sem LLM externo)",
        "",
        f"Sistema: {dossie['material']} medido por {dossie['tecnica']}.",
        f"Há {dossie['n_amostras_reais']} difratogramas reais (Nd = {dossie['dopagens_percent']} %).",
        "Os números abaixo vêm do pipeline Python, não de um modelo de linguagem.",
        "",
        "## Características",
    ]
    for rec in dossie["caracteristicas"]:
        linhas.append(
            f"- {rec['nd_percent']}% Nd: pico principal em {rec['two_theta_max']}°, "
            f"FWHM {rec['fwhm_main_deg']}°, n_picos={rec['n_peaks']}, "
            f"D(Scherrer)≈{rec.get('scherrer_D_nm')} nm, "
            f"D(WH)≈{rec.get('wh_D_nm')} nm, strain≈{rec.get('wh_strain')}."
        )
    linhas += ["", "## Machine Learning (Leave-One-Out)"]
    for nome, row in loo.items():
        linhas.append(
            f"- {nome}: R²_LOO={row['R2_LOO']}, MAE_LOO={row['MAE_LOO']} p.p., "
            f"RMSE_LOO={row['RMSE_LOO']} p.p. | R² de ajuste={row['R2_ajuste']} "
            f"(compare com o LOO para ver overfitting)."
        )
    syn = dossie["dados_sinteticos"]
    linhas += [
        "",
        "## Dados sintéticos",
        f"{syn['aviso']} R² no teste sintético = {syn['R2_teste_sintetico']:.3f} "
        f"(n_treino={syn['n_treino']}, n_teste={syn['n_teste']}).",
        "",
        "## Mensagem para a turma",
        "O valor científico está no DRX e nas características. O ML com n=5 é demonstração. "
        "O LLM, quando usado, só redige o que já foi calculado.",
    ]
    return "\n".join(linhas)


def _try_gemini(prompt: str, api_key: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return r.text
    except Exception:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text


def _try_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content


def get_secret(name: str):
    '''Colab Secrets primeiro; senao variavel de ambiente (uso local).'''
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def explicar_com_llm(dossie: dict) -> tuple[str, str]:
    payload = json.dumps(dossie, ensure_ascii=False, indent=2)
    prompt = SYSTEM + "\n\nJSON do pipeline:\n" + payload

    gem = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    oai = get_secret("OPENAI_API_KEY")

    if gem:
        try:
            return _try_gemini(prompt, gem), "gemini"
        except Exception as exc:
            print("Gemini indisponível:", exc)
    if oai:
        try:
            return _try_openai(prompt, oai), "openai"
        except Exception as exc:
            print("OpenAI indisponível:", exc)
    return relatorio_local(dossie), "local"


relatorio, fonte = explicar_com_llm(dossie)
print(f"Fonte da interpretação: {fonte}\n")
print(relatorio)
"""))

cells.append(md(r"""
### Segredo no Colab (opcional)

1. Ícone de chave (*Secrets*) → `GEMINI_API_KEY`
2. Marque **Notebook access** / acesso pelo notebook
3. Rode de novo a célula do copiloto

Chave gratuita: [Google AI Studio](https://aistudio.google.com/apikey). Sem chave, o relatório local já cobre a aula. Passo a passo no `docs/MANUAL.md`.
"""))

cells.append(md(r"""
---
# Módulo 7 — Pipeline híbrido completo

A célula abaixo reúne o fluxo em funções. A aula não foi uma sequência solta de gráficos: é um único caminho, do CSV ao relatório.

```
Difratograma (DRX)
        ↓
Leitura dos dados
        ↓
Pré-processamento
        ↓
Extração de características
        ↓
Machine Learning (Ridge + LOO)
        ↓
Avaliação do modelo
        ↓
LLM
        ↓
Interpretação científica
        ↓
Relatório final
```
"""))

cells.append(code(r"""
def run_hybrid_pipeline(dopings=DOPINGS):
    '''Uma funcao = o minicurso inteiro. Util para o aluno reexecutar depois de mudar um parametro.'''
    raw = {d: load_sample(d) for d in dopings}
    prep = {d: preprocess(df) for d, df in raw.items()}
    feat = pd.DataFrame([extract_features(d, prep[d]) for d in dopings])
    view = feat.drop(columns=["peak_positions", "peak_fwhm_deg"])
    X_loc = view[X_cols].to_numpy(dtype=float)
    y_loc = view[TARGET].to_numpy(dtype=float)
    col_m = np.nanmean(X_loc, axis=0)
    col_m = np.where(np.isfinite(col_m), col_m, 0.0)
    nan_at = np.where(np.isnan(X_loc))
    X_loc[nan_at] = np.take(col_m, nan_at[1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    y_loo = cross_val_predict(pipe, X_loc, y_loc, cv=LeaveOneOut())
    pipe.fit(X_loc, y_loc)

    resumo = {
        "amostras": dopings,
        "features": view.round(4).to_dict(orient="records"),
        "ridge_LOO": {
            "R2": float(r2_score(y_loc, y_loo)),
            "MAE": float(mean_absolute_error(y_loc, y_loo)),
            "RMSE": float(mean_squared_error(y_loc, y_loo) ** 0.5),
            "real": y_loc.tolist(),
            "predito": np.round(y_loo, 3).tolist(),
        },
    }
    texto, origem = explicar_com_llm({
        "material": "Bi1-xNdxFeO3",
        "tecnica": "DRX",
        "n_amostras_reais": len(dopings),
        "dopagens_percent": list(dopings),
        "caracteristicas": resumo["features"],
        "ml_loo": [{"modelo": "Ridge (pipeline final)", **{
            "R2_ajuste": float(r2_score(y_loc, pipe.predict(X_loc))),
            "MAE_ajuste": float(mean_absolute_error(y_loc, pipe.predict(X_loc))),
            "RMSE_ajuste": float(mean_squared_error(y_loc, pipe.predict(X_loc)) ** 0.5),
            "R2_LOO": resumo["ridge_LOO"]["R2"],
            "MAE_LOO": resumo["ridge_LOO"]["MAE"],
            "RMSE_LOO": resumo["ridge_LOO"]["RMSE"],
        }}],
        "previsoes_LOO": {"Ridge": {"real": y_loc.tolist(), "predito": resumo["ridge_LOO"]["predito"]}},
        "dados_sinteticos": dossie["dados_sinteticos"],
        "instrucoes_ao_llm": dossie["instrucoes_ao_llm"],
    })
    resumo["interpretacao_origem"] = origem
    resumo["relatorio"] = texto
    return prep, view, resumo


_prep, tabela_final, saida = run_hybrid_pipeline()
print("Origem da interpretação:", saida["interpretacao_origem"])
print("Ridge LOO:", {k: saida["ridge_LOO"][k] for k in ["R2", "MAE", "RMSE"]})
tabela_final.round(3)
"""))

cells.append(code(r"""
print(saida["relatorio"])
"""))

cells.append(md(r"""
## Encerramento

O minicurso percorreu:

1. o conceito de IA científica e a divisão de papéis (algoritmo vs. LLM);
2. dados reais de DRX de $\mathrm{Bi}_{1-x}\mathrm{Nd}_{x}\mathrm{FeO}_{3}$;
3. pré-processamento reproduzível;
4. características, Scherrer e Williamson–Hall (cristalito e strain);
5. quatro modelos de regressão, LOO, overfitting, sintéticos com ressalva;
6. LLM como redator/copiloto;
7. o pipeline inteiro em um único fluxo.

**Para levar para casa:** se o número não saiu de uma célula de cálculo, ele não entra no artigo — nem quando a frase do LLM estiver convincente.

Material de apoio: [`docs/MANUAL.md`](../docs/MANUAL.md) e [slides no navegador](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html).
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells, {OUT.stat().st_size} bytes)")
