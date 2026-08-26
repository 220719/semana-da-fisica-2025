# MC1 — Inteligência Artificial Científica aplicada à Difração de Raios X

> **Semana da Física 2026 — Universidade Estadual de Maringá (UEM)**  
> Ministrante: **Dr. Anuar José Mincache** (Pós-doutorado em Física — Lund University, Suécia)

Minicurso teórico-prático no **Google Colab**: pipeline verificável de DRX + Machine Learning + LLM como **copiloto** (interpretação e relatório). O modelo de linguagem **não substitui** a Física nem calcula 2θ, FWHM ou métricas.

**Sistema:** $\mathrm{Bi}_{1-x}\mathrm{Nd}_{x}\mathrm{FeO}_{3}$, $x$ = 10% … 50%.

---

## Abrir agora

| O quê | Clique aqui |
|---|---|
| **Slides (apresentação completa)** | [Abrir no navegador](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/apresentacao.html) |
| **Slides por módulo** | [Índice](https://raw.githack.com/220719/semana-da-fisica-2025/main/slides/index.html) |
| **Notebook da aula** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/220719/semana-da-fisica-2025/blob/main/notebooks/Minicurso_IA_Cientifica_DRX.ipynb) |
| Manual | [docs/MANUAL.md](docs/MANUAL.md) |

Não abra os arquivos `.html` na página de código do GitHub (`blob/main/slides/...`): isso mostra o fonte, não o slideshow. Use os links da tabela.

Setas do teclado avançam os slides; `F` é tela cheia.

---

## Comece por aqui (aula)

1. Abra o notebook no Colab pelo badge acima (CPU).
2. *Ambiente de execução → Executar tudo*.
3. Os CSV são lidos de `data/` ou baixados do GitHub. Não precisa montar o Drive nem instalar Python.

---

## Módulos

| # | Tema |
|---|---|
| 1 | O que é IA científica · ML × DL × LLM · pipeline híbrido |
| 2 | Arquivos de DRX, leitura correta, waterfall e mapa de intensidade |
| 3 | Savitzky–Golay, linha de base ALS, normalização |
| 4 | Picos, FWHM, Scherrer, Williamson–Hall (cristalito e strain), estatística |
| 5 | Linear, k-NN, SVR e Random Forest · LOO · overfitting |
| 6 | LLM como copiloto (Gemini/OpenAI opcional; fallback local) |
| 7 | Função única: do CSV ao relatório |

---

## Estrutura

```
├── notebooks/Minicurso_IA_Cientifica_DRX.ipynb
├── data/Nd_10.csv … Nd_50.csv
├── slides/          # Reveal.js, um HTML por módulo + apresentação
├── docs/MANUAL.md
├── tools/           # geradores do notebook e dos slides
├── requirements.txt
└── README.md
```

---

## Clone (alunos)

```bash
git clone https://github.com/220719/semana-da-fisica-2025.git
```

O nome da pasta no GitHub continua `semana-da-fisica-2025` (histórico do repositório). O minicurso é da **Semana da Física 2026**.

No Colab, o equivalente é uma célula `!git clone ...` — detalhes no manual.

---

## Correções em relação à edição anterior

- CSV **sem cabeçalho** (`header=None`): a versão antiga descartava o primeiro ponto.
- Sem dependência de Google Drive.
- Validação de ML por **Leave-One-Out** (com n = 5, split 80/20 não faz sentido).
- Scherrer **e** Williamson–Hall com β = FWHM de cada pico (a edição antiga usava um β da derivada da intensidade e gerava D ~ 0,1 nm).
- Gráficos: waterfall, mapa 2θ × dopagem, correlação/RMSE, histograma, boxplot/violin, WH, métricas em barras; PNGs em `figuras/` ao rodar o notebook.
- LLM só interpreta um JSON produzido pelo pipeline.

O notebook legado `notebooks/DRX_Analises.ipynb` (se presente) é histórico.

---

## Licença

[MIT](LICENSE)

## Autor

**Dr. Anuar José Mincache** · Semana da Física 2026 · UEM
