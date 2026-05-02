# 📘 MC1 — Pipeline para Análise de Difração de Raios X com Python e Machine Learning

> **Semana de Física 2025 — Universidade Estadual de Maringá (UEM)**

Mini-curso ministrado pelo **Dr. Anuar José Mincache** (Pós-doutorado em Física — Lund University, Suécia).

---

## 📌 Sobre o Projeto

Este repositório contém o material completo do mini-curso **MC1**, que apresenta a construção passo a passo de um pipeline para análise de padrões de **Difração de Raios X (DRX)** utilizando Python, Ciência de Dados e Machine Learning.

O objetivo é demonstrar como transformar arquivos brutos de difração em **insights estruturais**, dados organizados e modelos preditivos.

### Sistema Estudado

Amostras de **Bi₁₋ₓNdₓFeO₃** (Bismuto Ferrita dopada com Neodímio), com concentrações de Nd variando entre **10% e 50%**.

---

## 📂 Estrutura do Repositório

```
├── notebooks/
│   └── DRX_Analises.ipynb        # Notebook principal com todo o pipeline
├── data/
│   ├── Nd_10.csv                 # Difratograma — 10% Nd
│   ├── Nd_20.csv                 # Difratograma — 20% Nd
│   ├── Nd_30.csv                 # Difratograma — 30% Nd
│   ├── Nd_40.csv                 # Difratograma — 40% Nd
│   └── Nd_50.csv                 # Difratograma — 50% Nd
├── docs/
│   ├── Parte1.pdf                # Material de apoio — Parte 1
│   └── Parte2.pdf                # Material de apoio — Parte 2
├── requirements.txt              # Dependências do projeto
├── LICENSE                       # Licença MIT
└── README.md
```

---

## 🔬 Conteúdo do Mini-Curso

O notebook `DRX_Analises.ipynb` cobre as seguintes etapas:

1. **Carregamento e visualização** dos dados de DRX
2. **Detecção automática de picos** com `scipy.signal.find_peaks`
3. **Visualização 3D** dos difratogramas comparativos
4. **Análise estatística** — intensidade média, CV, barras de erro
5. **Heatmaps e correlação** entre picos e dopagens
6. **Tamanho de cristalito** — Equação de Scherrer e método de Williamson-Hall
7. **Machine Learning aplicado**:
   - Regressão Linear Simples
   - KNN Regressor
   - SVR (Support Vector Regression)
   - Regressão Polinomial (grau 2)
   - Random Forest Regressor
8. **Data Augmentation** — geração de dados sintéticos para melhoria dos modelos

---

## 🚀 Como Usar

### Opção 1 — Google Colab (recomendado)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/220719/anuarmincache/blob/main/notebooks/DRX_Analises.ipynb)

1. Clique no badge acima para abrir no Google Colab
2. Faça upload dos arquivos `.csv` da pasta `data/` ou monte seu Google Drive
3. Execute as células sequencialmente

### Opção 2 — Ambiente Local

```bash
git clone https://github.com/220719/anuarmincache.git
cd anuarmincache
pip install -r requirements.txt
jupyter notebook notebooks/DRX_Analises.ipynb
```

---

## 📦 Dependências

| Pacote | Uso |
|--------|-----|
| `pandas` | Manipulação de dados |
| `numpy` | Cálculos numéricos |
| `matplotlib` | Visualização de gráficos |
| `seaborn` | Gráficos estatísticos |
| `scipy` | Detecção de picos e análise de sinais |
| `scikit-learn` | Modelos de Machine Learning |

---

## 📊 Dados

Os arquivos CSV contêm duas colunas:

| Coluna | Descrição |
|--------|-----------|
| `2θ (°)` | Ângulo de difração |
| `Intensidade (u.a.)` | Intensidade do sinal difratado |

---

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE).

---

## 👤 Autor

**Dr. Anuar José Mincache**

- Pós-doutorado em Física — Lund University (Suécia)
- Semana de Física 2025 — UEM

---

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.
